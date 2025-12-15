# with system prompt v0.1.0 (16 Dec 2025)

- does not follow all instructions properly, e.g. for query id 2, did not provide
estimate of preparation time.
- even if it does provide, it is not standardised; sometimes at the front, sometimes
at the end. this could be because in the system prompt, I didn't specify in the example
output where it should be. I can fix this.
    * However, it also raises a question, do we need another LLM pass to standardise
    the format (if it didn't adhere to it perfectly?)
    * This requires us to write some parser to check.
    * I guess overkill for now, but necessary if we want to store the responses properly
