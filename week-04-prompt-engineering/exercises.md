# Week 4 exercises: Prompt Engineering

Run the experiment once and save your summary. Change one variable at a time.

1. **Inspect errors:** Find three tickets each version misclassified. Which category rule would help clarify them?
2. **Compare prompts:** Remove the paid-plan exception from the structured instructions. Does ticket T019 change?
3. **Few-shot examples:** Replace one few-shot example with another example **outside** the evaluation dataset. Does accuracy improve or decline?
4. **Output validity:** Compare exact-label compliance for the free-text versions against schema-enforced output. Why is formatting different from classification correctness?
5. **Token usage:** Sum `input_tokens` and `output_tokens` by version. Which configuration uses the most tokens?
6. **Repetition:** Run the same evaluation again with the same model. Which outputs changed? Does `temperature=0` guarantee identical results?
7. **Error analysis:** Add a fifth label only after updating the ground truth and category definitions. Why can't you fairly compare the new five-category task against the original four-category accuracy?

**Reflection:** Explain why a more detailed prompt can improve reliability without changing the model's weights, and why a structured output schema does not guarantee a correct answer.
