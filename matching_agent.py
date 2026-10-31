from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).parent
RESUMES_DIR = ROOT / "resumes"


class AgentState(TypedDict, total=False):
	conversation: list[dict[str, str]]
	job_description: str
	requirements: dict[str, list[str]]
	resumes: list[dict[str, str]]
	rankings: list[dict[str, Any]]
	report: str
	feedback: str


def _json_from_response(response: Any) -> dict[str, Any]:
	content = response.content if hasattr(response, "content") else str(response)
	if not isinstance(content, str):
		content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
	match = re.search(r"\{.*\}", content, re.DOTALL)
	if not match:
		raise ValueError(f"Model did not return JSON: {content[:200]}")
	return json.loads(match.group(0))


class ProfileMatchingAgent:

	def __init__(self, resumes_dir: Path = RESUMES_DIR) -> None:
		load_dotenv(ROOT / ".env")
		api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
		if not api_key:
			raise RuntimeError("Set OPEN_ROUTER_KEY in .env before starting the agent.")

		self.resumes_dir = resumes_dir
		self.model = ChatOpenAI(
			model=os.getenv("OPEN_ROUTER_MODEL", "openrouter/free"),
			api_key=api_key,
			base_url="https://openrouter.ai/api/v1",
			temperature=0,
			default_headers={
				"HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
				"X-Title": os.getenv("OPENROUTER_SITE_NAME", "Profile Matching Agent"),
			},
		)
		workflow = StateGraph(AgentState)
		workflow.add_node("parse_jd", self.parse_jd)
		workflow.add_node("extract_requirements", self.extract_requirements)
		workflow.add_node("search_resumes", self.search_resumes)
		workflow.add_node("rank_candidates", self.rank_candidates)
		workflow.add_node("generate_report", self.generate_report)
		workflow.add_edge(START, "parse_jd")
		workflow.add_edge("parse_jd", "extract_requirements")
		workflow.add_edge("extract_requirements", "search_resumes")
		workflow.add_edge("search_resumes", "rank_candidates")
		workflow.add_edge("rank_candidates", "generate_report")
		workflow.add_edge("generate_report", END)
		self.workflow = workflow.compile()

	def parse_jd(self, state: AgentState) -> AgentState:
		feedback = state.get("feedback", "")
		job_description = state["job_description"]
		conversation = list(state.get("conversation", []))
		if feedback:
			conversation.append({"role": "user", "content": feedback})
		if feedback:
			job_description = f"{job_description}\nAdditional user criteria: {feedback}"
		return {"job_description": job_description, "conversation": conversation}

	def extract_requirements(self, state: AgentState) -> AgentState:
		prompt = """Extract hiring requirements from the job description.
Return only JSON with this shape: {"must_have": ["..."], "nice_to_have": ["..."]}.
Keep requirements concrete and concise. Do not invent requirements.

JOB DESCRIPTION:
""" + state["job_description"]
		result = _json_from_response(self.model.invoke([HumanMessage(content=prompt)]))
		return {
			"requirements": {
				"must_have": [str(item) for item in result.get("must_have", [])],
				"nice_to_have": [str(item) for item in result.get("nice_to_have", [])],
			}
		}

	def search_resumes(self, state: AgentState) -> AgentState:
		resumes = []
		for path in sorted(self.resumes_dir.glob("*.txt")):
			resumes.append({"id": path.stem, "name": path.stem.replace("resume_", "").replace("_", " ").title(), "text": path.read_text()})
		if not resumes:
			raise FileNotFoundError(f"No .txt resumes found in {self.resumes_dir}")
		return {"resumes": resumes}

	def rank_candidates(self, state: AgentState) -> AgentState:
		prompt = """Rank every candidate against the requirements. Return only JSON:
{"rankings": [{"candidate_id": "resume_file_stem", "score": 0, "strengths": [], "gaps": [], "reasoning": ""}]}
Score from 0 to 100. Use evidence from the resume, distinguish missing evidence from a definite gap,
and include every candidate exactly once, sorted best first.

REQUIREMENTS:
""" + json.dumps(state["requirements"]) + "\n\nCANDIDATES:\n" + json.dumps(state["resumes"])
		result = _json_from_response(self.model.invoke([HumanMessage(content=prompt)]))
		return {"rankings": result.get("rankings", [])}

	def generate_report(self, state: AgentState) -> AgentState:
		top = state["rankings"][:10]
		prompt = """Write a concise hiring shortlist report in Markdown.
Include the requirements, a ranked table with score, strengths, and gaps, then a short recommendation.
Do not claim facts that are absent from the resume evidence.

REQUIREMENTS:
""" + json.dumps(state["requirements"]) + "\nRANKINGS:\n" + json.dumps(top)
		response = self.model.invoke([SystemMessage(content="You are a careful, evidence-based recruiter."), HumanMessage(content=prompt)])
		return {"report": response.content}

	def run(self, job_description: str, feedback: str = "") -> AgentState:
		conversation = [{"role": "user", "content": job_description}]
		if feedback:
			conversation.append({"role": "user", "content": feedback})
		return self.workflow.invoke({"job_description": job_description, "feedback": feedback, "conversation": conversation})

	def refine(self, state: AgentState, feedback: str) -> AgentState:
		"""Re-run matching after the user changes the hiring criteria."""
		return self.workflow.invoke(
			{
				"job_description": state["job_description"],
				"feedback": feedback,
				"conversation": state.get("conversation", []),
			}
		)

	def compare_candidates(self, state: AgentState, candidate_ids: list[str]) -> str:
		selected = [item for item in state.get("rankings", []) if item.get("candidate_id") in candidate_ids]
		if len(selected) < 2:
			raise ValueError("Provide at least two candidate IDs from the ranking results.")
		prompt = "Compare these candidates side by side using only the supplied evidence. End with a recommendation.\n" + json.dumps(selected)
		return self.model.invoke([HumanMessage(content=prompt)]).content

	def generate_interview_questions(self, state: AgentState, candidate_id: str) -> str:
		candidate = next((item for item in state.get("rankings", []) if item.get("candidate_id") == candidate_id), None)
		if not candidate:
			raise ValueError(f"Unknown candidate: {candidate_id}")
		prompt = "Generate five targeted screening questions for this candidate. Include what each question validates.\n" + json.dumps(candidate)
		return self.model.invoke([HumanMessage(content=prompt)]).content


def main() -> None:
	job_description = input("Enter the job description: ").strip()
	if not job_description:
		raise ValueError("A job description is required.")
	agent = ProfileMatchingAgent()
	state = agent.run(job_description)
	print(state["report"])
	print("\nCandidate IDs: " + ", ".join(item["candidate_id"] for item in state["rankings"]))


if __name__ == "__main__":
	main()
