import dspy

lm = dspy.LM("databricks/databricks-meta-llama-3-1-70b-instruct")
dspy.configure(lm=lm)
