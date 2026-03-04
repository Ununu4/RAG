# Faithfulness Verification Report

## Sample Answer (Pharmacy, Iowa, 60K, 3 positions, $100K/mo)

| Claim | Guidelines Check | Verdict |
|-------|------------------|---------|
| **Cashable**: $25k/mo min, 2nd-10th, IA allowed | cashable_guidelines: Positions 2nd-10th ✓, Min $25,000 ✓, IA not in restricted (HI/AK/PR/CA) ✓ | ✅ **CORRECT** |
| **Aspire**: 1st-6th position, $100K/mo revenue, IA allowed | aspire: 1st-6th ✓, IA not restricted ✓. **$100K/mo** – Aspire has NO minimum revenue in guidelines; says "Net 30-50%". Phrasing implies requirement. | ⚠️ **MISLEADING** |
| **Aurum**: 2nd-10th, $30k/mo min, IA allowed | aurum: 2nd-10th ✓, $30K min ✓, no state restrictions ✓ | ✅ **CORRECT** |
| **BizFund**: 2nd-10th position, IA allowed | bizfund: **Positions 1st-3rd** (NOT 2nd-10th). Auto decline 4+ positions. User has 3 positions = 4th for new = **AUTO DECLINE**. | ❌ **WRONG** |
| **501 Advance**: $20k/mo min, 1 year TIB, IA allowed | 501_advance: $20K min ✓, 1yr TIB ✓, IA not in restricted (CA/HI/PR) ✓ | ✅ **CORRECT** |

## Metric Assessment

### Faithfulness Score: 1.0 (Reported) vs Ground Truth

- **NLI passed all sentences** – citation-aware context + 0.45 threshold
- **Ground truth**: 2 errors (BizFund position wrong + fit wrong; Aspire revenue phrasing)
- **Conclusion**: Faithfulness metric may be **over-optimistic** – NLI entailment does not catch:
  1. **Numerical contradictions** (2nd-10th vs 1st-3rd)
  2. **Implied requirements** not in source ($100K/mo for Aspire)
  3. **Fit logic** (3 positions → 4th for new → BizFund auto-decline)

### Preventive Measures (Implemented)

1. **Position logic in prompt**: When user has N positions, prompt explicitly states: "User has N positions = (N+1)th for new funding. EXCLUDE lenders that auto-decline (N+1)+ or only accept 1st through Nth."
2. **Quote-first instruction**: "COPY the exact position and revenue requirements from the source—do not paraphrase or infer."
3. **Retrieval enrichment**: When user has positions, search query includes "positions auto decline eligibility" to surface chunks with position rules.
4. **Schema hint**: Emphasizes "copy exact positions + revenue from source" to discourage hallucination.
