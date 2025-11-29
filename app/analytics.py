# app/analytics_utils.py

from sqlalchemy.orm import Session
from sklearn.cluster import KMeans
from .models import QueryLog, Agent
from .rag import provider_from
import numpy as np


def fetch_queries(db: Session, agent_id: int):
    logs = (
        db.query(QueryLog)
        .filter_by(agent_id=agent_id)
        .order_by(QueryLog.timestamp.asc())
        .all()
    )
    return [log.query for log in logs]

def cluster_queries(queries, embeddings, k=5):
    if len(queries) == 0:
        return []

    k = min(k, len(queries))
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for idx, label in enumerate(labels):
        label = int(label)  # ← FIX: convert numpy.int32 to python int
        clusters.setdefault(label, []).append(queries[idx])

    result = []
    for cid, qlist in clusters.items():
        result.append({
            "cluster_id": int(cid),        # ← FIX
            "count": int(len(qlist)),      # ← FIX
            "representative": qlist[0],
            "questions": list(qlist[:10])  # ensure it's a list, not numpy object
        })
    return result



def summarize_clusters(clusters, provider):
    """
    Send cluster info to LLM for summarization.
    """
    if not clusters:
        return ""

    text = "Here are clusters of student questions:\n\n"
    for c in clusters:
        text += f"Cluster {c['cluster_id']} (Count: {c['count']}):\n"
        text += f"Representative: {c['representative']}\n"
        text += "Example Questions:\n"
        for q in c["questions"][:5]:
            text += f"- {q}\n"
        text += "\n"

    prompt = [
        {
            "role": "system",
            "content": (
                "You are analyzing student questions for a university course. "
                "Your job is to generate a clear and concise FAQ-style summary. "

                "### REQUIRED FORMAT (FOLLOW EXACTLY):\n"
                "For each FAQ theme, output:\n"
                "\n"
                "Theme Title\n"
                "- Number of Questions: X\n"
                "- Students Concern: <very short explanation>\n"
                "\n"
                "FORMAT RULES:\n"
                "- NO blank line between the theme title and its bullet points.\n"
                "- Leave ONE blank line **between different sections only**.\n"
                "- The context sentence must start with: 'Students are asking about ...'\n"
                "- Keep context short (1 sentence).\n"
                "- Keep the tone objective.\n"
                "- You may provide teaching advice only if explicitly relevant.\n"
                "- NO markdown formatting symbols like ### or **.\n"
            )
        },
        {"role": "user", "content": text}
    ]



    try:
        summary = provider.complete(prompt)
        return summary.strip()
    except Exception as e:
        print("[Analytics Summarization Error]", e)
        return "Error summarizing clusters."
