from rag_runner import run_rag_pipeline

payload = {
    "documents": [
        "data/BAJHLIP23020V012223.pdf",
        "data/CHOTGDP23004V012223.pdf",
        "data/EDLHLGA23009V012223.pdf",
        "data/HDFHLIP23024V072223.pdf",
        "data/ICIHLIP22012V012223.pdf",
        "data/policy.pdf"
    ],
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Run RAG pipeline
results = run_rag_pipeline(payload["documents"], payload["questions"])

# Display answers
print("\nFinal Answers:\n")
for item in results:
    print(f"{item['question']}")
    print(f"{item['answer']}\n")
