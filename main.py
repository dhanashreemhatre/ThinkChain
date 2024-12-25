from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


# Initialize the LLaMA model via Ollama
llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")  

# Step-by-Step Prompt Template
cot_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are solving a problem step-by-step. Each step must be accurate, "
        "and the reasoning must be clear and concise. \n"
        "For every computation, explicitly mention the steps taken. \n"
        "Question: {question}\n"
        "Answer (step-by-step):"
    ),
)

# Refinement Prompt Template
refinement_prompt = PromptTemplate(
    input_variables=["intermediate_answer"],
    template=(
        "Review the following reasoning and result carefully:\n"
        "Reasoning so far:\n{intermediate_answer}\n"
        "Verify all steps logically and re-compute if necessary. "
        "Provide a corrected final answer if needed and explain why it is correct."
    ),
)

# Validation Prompt Template
validation_prompt = PromptTemplate(
    input_variables=["final_answer"],
    template=(
        "Final verification:\n"
        "{final_answer}\n"
        "Double-check the computation, verify the reasoning, and confirm if the answer is indeed correct. "
        "If correct, respond 'Verified: [answer]'. If not, explain the error and correct it."
    ),
)

# Chain for Initial Reasoning
cot_chain = LLMChain(llm=llm, prompt=cot_prompt, output_key="intermediate_answer")

# Chain for Refinement
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt, output_key="final_answer")

# Chain for Validation
validation_chain = LLMChain(llm=llm, prompt=validation_prompt, output_key="verified_answer")

# Combine Chains into a Sequential Chain
full_cot_chain = SequentialChain(
    chains=[cot_chain, refinement_chain, validation_chain],

    input_variables=["question"],  # Input to the first chain
    output_variables=["verified_answer"],  # Final output from the chain
)

# Function to Run the Full Chain
def iterative_cot_response(question: str):
    """
    Generate a refined Chain of Thought response using iterative reasoning.
    """
    return full_cot_chain.run({"question": question})

# Example Usage
if __name__ == "__main__":
    question = "Count the number of 'R's in the given string 'Strawberry'."
    response = iterative_cot_response(question)

    print("Iterative Chain of Thought Response:")
    print(response)
