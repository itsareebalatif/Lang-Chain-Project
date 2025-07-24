from typing import Dict, Any, cast
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Create model
model = ChatGroq(model="llama3-70b-8192")

# Define templates with clearer instructions
positive_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Write a warm thank you response to positive feedback."),
    ("human", "Feedback: {feedback}")
])

negative_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Write a helpful response to address negative feedback. Offer solutions, not escalation."),
    ("human", "Feedback: {feedback}")
])

neutral_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Politely ask for more details about neutral feedback."),
    ("human", "Feedback: {feedback}")
])

escalate_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Create a message to escalate this case to a human."),
    ("human", "Feedback: {feedback}")
])

# More precise classification template
classification_template = ChatPromptTemplate.from_messages([
    ("system", """Classify this feedback EXACTLY as:
- 'positive' if happy or satisfied
- 'negative' if unhappy but doesn't request human
- 'escalate' if specifically asks for manager/human
- 'neutral' if neither positive nor negative

Return ONLY the classification word."""),
    ("human", "{feedback}")
])

def classify_feedback(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Get classification
    classification = (classification_template | model | StrOutputParser()).invoke(input_dict)
    classification = classification.lower().strip()
    
    # Force negative classification unless explicit escalation
    if "escalate" not in classification and "manager" not in input_dict["feedback"].lower():
        if "terrible" in input_dict["feedback"].lower() or "broke" in input_dict["feedback"].lower():
            classification = "negative"
    
    return {
        "sentiment": classification,
        "feedback": input_dict["feedback"]
    }

# Branch with explicit checks and type casting
branches = RunnableBranch(
    (lambda x: cast(Dict[str, Any], x)["sentiment"] == "positive", 
     positive_template | model | StrOutputParser()),
    
    (lambda x: cast(Dict[str, Any], x)["sentiment"] == "negative", 
     negative_template | model | StrOutputParser()),
    
    (lambda x: cast(Dict[str, Any], x)["sentiment"] == "neutral", 
     neutral_template | model | StrOutputParser()),
    
    (escalate_template | model | StrOutputParser())
)

# Final chain
chain = (
    RunnableLambda(classify_feedback)
    | branches
)

# Test with negative feedback
review = "The product is very good .i like it "
result = chain.invoke({"feedback": review})
print("Customer Service Response:")
print(result)