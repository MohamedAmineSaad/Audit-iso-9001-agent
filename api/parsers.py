from __future__ import annotations
import re
from typing import Any, Dict, List


def parse_stats(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # match rows like: │ Collection          │ iso_9001_knowledge │
    rows = re.findall(r"│\s*(.+?)\s*│\s*(.+?)\s*│", stdout)
    for k, v in rows:
        kk = k.strip().lower()
        vv = v.strip()
        if kk.startswith("collection"):
            out["collection"] = vv
        elif kk.startswith("documents"):
            try:
                out["documents_indexed"] = int(vv)
            except:
                out["documents_indexed"] = None
        elif "embeddings" in kk:
            out["embedding_model"] = vv
        elif "llm" in kk:
            out["llm_model"] = vv
        elif kk.startswith("répertoire") or kk.startswith("repertoire"):
            out["vectorstore_path"] = vv
    return out


def parse_search(stdout: str, query: str, k: int) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    # Example header: 1. ISO_9000_2015 (Similarité: 46.18%)
    # Then: Page 37 | Section 3
    blocks = re.split(r"\n(?=\d+\.\s)", stdout)
    for b in blocks:
        m = re.match(r"(\d+)\.\s+(.+?)\s+\(Similarité:\s*([\d.]+)%\)", b.strip())
        if not m:
            continue
        rank = int(m.group(1))
        doc = m.group(2).strip()
        sim = float(m.group(3))
        page = None
        section = None
        mp = re.search(r"Page\s+(\d+)\s+\|\s+Section\s+([^\n]+)", b)
        if mp:
            page = int(mp.group(1))
            section = mp.group(2).strip()
        # excerpt = everything after that line
        excerpt = b
        # remove first lines
        excerpt = re.sub(r"^\d+\..*?\n", "", excerpt, flags=re.DOTALL).strip()
        hits.append(
            {
                "rank": rank,
                "document": doc,
                "similarity_percent": sim,
                "page": page,
                "section": section,
                "excerpt": excerpt.strip(),
            }
        )
    return {"query": query, "k": k, "hits": hits}


def parse_chat(stdout: str) -> Dict[str, Any]:
    # Extract answer between "📋 Réponse" and "📚 Sources:"
    answer = ""
    sources: List[Dict[str, Any]] = []

    # Answer box content: take everything after "📋 Réponse" box line markers
    if "📚 Sources:" in stdout:
        before_sources = stdout.split("📚 Sources:")[0]
    else:
        before_sources = stdout

    # get answer text inside box: easiest: remove box borders and take lines after "📋 Réponse"
    if "📋 Réponse" in before_sources:
        chunk = before_sources.split("📋 Réponse", 1)[1]
        # remove box characters
        lines = []
        for line in chunk.splitlines():
            # remove leading box drawing chars
            line2 = line.replace("│", "").strip()
            if line2 and not line2.startswith("╭") and not line2.startswith("╰"):
                lines.append(line2)
        answer = "\n".join(lines).strip()

    # Parse sources numbered [1] ...
    if "📚 Sources:" in stdout:
        s_part = stdout.split("📚 Sources:", 1)[1]
        # each source begins with [n]
        items = re.split(r"\n\s*\[(\d+)\]\s*", "\n" + s_part)
        # items looks like: ["", "1", "ISO_9001...", "2", ...]
        for i in range(1, len(items), 2):
            sid = int(items[i])
            body = items[i + 1]
            first = body.strip().splitlines()[0].strip()
            # first line example: ISO_9001_2015 - Section 0.4
            doc = first
            sec = None
            mm = re.match(r"(.+?)\s*-\s*Section\s*(.+)", first)
            if mm:
                doc = mm.group(1).strip()
                sec = mm.group(2).strip()

            page = None
            title = None
            mpt = re.search(r"Page\s+(\d+)\s*\|\s*(.+)", body)
            if mpt:
                page = int(mpt.group(1))
                title = mpt.group(2).strip()

            # excerpt: take last non-empty paragraph-ish lines (short)
            excerpt_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            excerpt = ""
            if len(excerpt_lines) >= 4:
                excerpt = " ".join(excerpt_lines[-4:])[:500]
            else:
                excerpt = " ".join(excerpt_lines)[:500]

            sources.append(
                {
                    "id": sid,
                    "document": doc,
                    "section": sec,
                    "page": page,
                    "title": title,
                    "excerpt": excerpt,
                }
            )

    # question (optional) — keep empty if not found
    q = ""
    mq = re.search(r"Question:\s*(.+)", stdout)
    if mq:
        q = mq.group(1).strip()

    return {"question": q, "answer": answer, "sources": sources}


def parse_analyze_summary(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"priority_actions": []}

    # doc metadata table
    mf = re.search(r"Nom du fichier\s*│\s*(.+?)\s*│", stdout)
    if mf:
        out["file_name"] = mf.group(1).strip()
    mt = re.search(r"Type\s*│\s*(.+?)\s*│", stdout)
    if mt:
        out["file_type"] = mt.group(1).strip()
    ms = re.search(r"Nombre de sections\s*│\s*(\d+)\s*│", stdout)
    if ms:
        out["sections_count"] = int(ms.group(1))

    # recap table
    mg = re.search(r"Score de conformité global\s*│\s*([\d.]+)%", stdout)
    if mg:
        out["global_score_percent"] = float(mg.group(1))
    mtc = re.search(r"Total de clauses analysées\s*│\s*(\d+)", stdout)
    if mtc:
        out["total_clauses_analyzed"] = int(mtc.group(1))
    mc = re.search(r"Clauses conformes\s*│\s*(\d+)", stdout)
    if mc:
        out["clauses_conformes"] = int(mc.group(1))
    mpc = re.search(r"Clauses partiellement conformes\s*│\s*(\d+)", stdout)
    if mpc:
        out["clauses_partiellement_conformes"] = int(mpc.group(1))
    mnc = re.search(r"Clauses non conformes\s*│\s*(\d+)", stdout)
    if mnc:
        out["clauses_non_conformes"] = int(mnc.group(1))

    # priority actions list after "🎯 Actions prioritaires:"
    if "🎯 Actions prioritaires:" in stdout:
        tail = stdout.split("🎯 Actions prioritaires:", 1)[1]
        for line in tail.splitlines():
            line = line.strip()
            if line.startswith("•"):
                out["priority_actions"].append(line[1:].strip())

    return out


def parse_agentic(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"priority_actions": [], "sections": []}

    # doc info table
    mf = re.search(r"Nom du fichier\s*│\s*(.+?)\s*│", stdout)
    if mf:
        out["file_name"] = mf.group(1).strip()
    mt = re.search(r"Type\s*│\s*(.+?)\s*│", stdout)
    if mt:
        out["file_type"] = mt.group(1).strip()
    ms = re.search(r"Sections\s*│\s*(\d+)\s*│", stdout)
    if ms:
        out["sections_total"] = int(ms.group(1))

    # global report table
    man = re.search(r"Sections analysées\s*│\s*(\d+)", stdout)
    if man:
        out["sections_analyzed"] = int(man.group(1))
    mav = re.search(r"Score de conformité moyen\s*│\s*([\d.]+)%", stdout)
    if mav:
        out["average_score_percent"] = float(mav.group(1))
    mh = re.search(r"Sections haute confiance\s*│\s*(\d+)", stdout)
    if mh:
        out["sections_high_confidence"] = int(mh.group(1))
    mm = re.search(r"Sections confiance moyenne\s*│\s*(\d+)", stdout)
    if mm:
        out["sections_medium_confidence"] = int(mm.group(1))
    ml = re.search(r"Sections faible confiance\s*│\s*(\d+)", stdout)
    if ml:
        out["sections_low_confidence"] = int(ml.group(1))

    # per-section summaries: capture blocks starting with "🔍 AGENTIC ANALYSIS:"
    sec_blocks = re.split(r"\n(?=Section \d+/\d+: )", stdout)
    for b in sec_blocks:
        mtitle = re.search(r"Section \d+/\d+:\s*(.+)", b)
        if not mtitle:
            continue
        title = mtitle.group(1).strip()

        # confidence line: "⚠️ Confiance: MEDIUM (0.65)"
        conf_label = None
        conf_score = None
        mc = re.search(r"Confiance:\s*([A-Z]+)\s*\(([\d.]+)\)", b)
        if mc:
            conf_label = mc.group(1).strip()
            conf_score = float(mc.group(2))

        mscore = re.search(r"Score de conformité:\s*(\d+)%", b)
        score = int(mscore.group(1)) if mscore else None

        # "Qualité: Récupération=insufficient, Preuves=moderate"
        retrieval_quality = None
        evidence_quality = None
        mq = re.search(r"Qualité:\s*Récupération=([^,]+),\s*Preuves=([^\n]+)", b)
        if mq:
            retrieval_quality = mq.group(1).strip()
            evidence_quality = mq.group(2).strip()

        # risks list:
        risks = []
        if "⚠️ Risques identifiés:" in b:
            tail = b.split("⚠️ Risques identifiés:", 1)[1]
            for line in tail.splitlines():
                line = line.strip()
                if line.startswith("•"):
                    risks.append(line[1:].strip())

        # clauses:
        clauses = []
        if "Analysé" in b and "clauses ISO" in b:
            for line in b.splitlines():
                line = line.strip()
                if "Clause" in line and ":" in line and "%" in line:
                    clauses.append(line.replace("•", "").strip())

        out["sections"].append(
            {
                "title": title,
                "confidence_label": conf_label,
                "confidence_score": conf_score,
                "conformity_score_percent": score,
                "retrieval_quality": retrieval_quality,
                "evidence_quality": evidence_quality,
                "risks": risks,
                "clauses": clauses,
            }
        )

    # priority actions at end (agentic uses risks as actions)
    if "🎯 Actions prioritaires:" in stdout:
        tail = stdout.split("🎯 Actions prioritaires:", 1)[1]
        for line in tail.splitlines():
            line = line.strip()
            if line.startswith("•"):
                out["priority_actions"].append(line[1:].strip())

    return out
