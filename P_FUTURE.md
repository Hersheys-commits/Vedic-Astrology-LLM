# Future Scope

## Near-term evaluation follow-up

- Validate both local Qwen3-4B LoRA paths separately on target hardware. The OpenVINO NPU path has returned visible output in the frontend and has a configured 2,048-token ceiling, but peak memory, latency and answer completeness at that limit still need measurement. Full Transformers CPU generation remains unverified even though a small bitsandbytes CPU forward-pass smoke check passed. Measure peak memory, load time, generation latency and answer quality before calling either path production-ready.
- Review the latest development and holdout retrieval reports, including every failed case and its cited PDF page. Do not accept retrieval changes solely because an automated score increased.
- Confirm `OPENAI_API_KEY` is set privately and configure a separate `EVAL_JUDGE_MODEL`, then run answer generation on development cases first. Review answers, citations, judge rationales and token usage against source pages before changing prompts or labels.
- Run the locked holdout after development changes are selected and use it to check regressions. The current suite is synthetic and exploratory; ask a knowledgeable Jyotisha reviewer to adjudicate evidence and interpretations before treating it as a quality benchmark.
- Keep unsupported and high-stakes cases in a separate safety review. A judge score is diagnostic and does not validate an astrological conclusion.

## From Prototype to Reliable Vedic Astrology Assistant

### Long-term objective

The long-term goal is to build an assistant that can interpret a user's Vedic chart in response to natural-language questions, while making clear which calculated chart facts and textual sources informed each answer. The system should communicate uncertainty, acknowledge missing or conflicting evidence, and present astrology as a traditional interpretive framework rather than a guarantee of future events.

An optional fine-tuned Qwen3-4B LoRA adapter is now integrated for local answer generation. It is separate from the planned project-trained model of approximately 1 billion parameters, which has not yet been trained. The aim is to determine through evaluation whether a compact model can handle domain-specific questions and produce useful, grounded responses at an acceptable quality and cost.

### Proposed development stages

#### 1. Strengthen the chart and corpus foundations

- Review chart calculations against carefully selected reference charts and document the supported calculation conventions.
- Make time-dependent chart details reproducible by recording the effective calculation date and settings.
- Audit the OCR, chapter coverage, and source metadata of the existing books; correct errors where they can be verified.
- Document source editions and permissions, and add further sources only when their provenance and permitted use are clear.
- Improve retrieval based on observed errors—for example, cases where the right topic is found but the wrong rule or source passage ranks highly.

**Intended result:** A more dependable calculation and evidence base before training or expanding the language model.

#### 2. Create evaluation datasets

Develop separate, reviewed examples for each stage of the system:

- **Chart calculation:** birth inputs and settings paired with reference chart values.
- **Question understanding:** questions paired with the chart facts needed to answer them.
- **Retrieval:** questions and chart facts paired with relevant source passages and provenance.
- **Answer quality:** evidence and chart context paired with acceptable answers, citations, and uncertainty behavior.
- **Safety and robustness:** examples involving missing evidence, conflicting texts, ambiguous questions, prompt injection, or requests for certain predictions.

Qualified reviewers should adjudicate difficult source interpretations. Evaluation examples should be separated from training examples so a model is not assessed on material it has already seen in near-identical form.

**Intended result:** Measurable baselines that help identify whether failures come from calculation, question parsing, retrieval, or answer generation.

#### 3. Add layered safeguards and reliability checks

- Validate incoming birth details, chart structure, and supported calculation settings.
- Check that extracted intents and facts use supported values and correspond to the supplied chart.
- Detect when retrieval returns little relevant evidence, when source coverage is incomplete, or when retrieved passages conflict.
- Require answer claims to refer to retrieved passage IDs where appropriate, and verify that cited passages were actually returned.
- Make the answer acknowledge uncertainty or abstain when evidence is weak; avoid presenting outcomes as certain.
- Add privacy and security controls, such as restricted API access, rate limits, careful handling of birth data, and protection against instructions embedded in user text or source documents.

These controls can reduce errors but cannot make an interpretive system completely error-free. They should be tested as part of every release.

#### 4. Prepare a high-quality training dataset

Before fine-tuning, create a reviewed dataset that links each example to the information needed to reproduce and assess it: question, chart context and calculation settings, retrieved passages, source references, example answer, uncertainty or refusal labels, and review/provenance details.

Training material should be based on source content that the project is permitted to use. OCR text and model-generated answers should not be treated as verified training truth without review. Private user charts should not be included unless there is informed consent and appropriate data handling. Examples should include both straightforward questions and difficult cases, such as irrelevant evidence, missing chart features, contradictory rules, and unsupported requests.

**Intended result:** A traceable dataset designed to teach response behavior and evidence use, rather than simply encourage the model to make confident predictions.

#### 5. Evaluate a compact custom language model

Select an open-weight model near 1B parameters after reviewing its license, language support, context size, and suitability for the intended hardware. Compare it with both an unmodified version of the candidate model and the current hosted-model baseline.

If the data and evaluation results justify training, begin with supervised fine-tuning using a parameter-efficient method such as LoRA or QLoRA. Fine-tuning should focus on question handling, terminology, answer structure, citation behavior, and appropriate uncertainty. Chart calculation and source retrieval should remain separate system components, so numeric chart facts and updated source material do not have to be memorized by the model.

The model should only be considered for use if it performs adequately on held-out examples for chart-fact fidelity, evidence use, citation correctness, uncertainty, and safety. Its speed and memory use should also be measured on the hardware where it is expected to run.

**Intended result:** Evidence about whether a compact, domain-adapted model is practical—not an assumption that a smaller model will automatically perform as well as a larger hosted model.

#### 6. Expand features after validation

Once the core system has reliable benchmarks and safeguards, the project can consider additional validated chart calculations, broader source coverage, additional languages, and more useful user-facing ways to inspect evidence. Each new capability should have its own test and review examples before being presented as supported.

### Expected outcome

The future system should combine reproducible chart calculation, traceable textual evidence, careful language generation, and measurable quality controls. The main research question is whether this combination—potentially supported by a fine-tuned ~1B model—can produce useful, transparent answers while keeping calculation and doctrinal evidence inspectable.
