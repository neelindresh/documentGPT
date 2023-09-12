def extract(text,exception):
    if exception in text:
        idx=text.index(exception)
        text=text[:idx]
    return text
     
def post_processcor(text):
    text=text.replace("<|im_end|>",'')
    text=extract(text,"Question: ")
    text=extract(text,"Use the following pieces of context to answer the question ")
    text=extract(text,"``` ")
    text=extract(text,'``` ')
    text=extract(text," ``` ##")
    text=extract(text,"print(")    
    text=extract(text,"Unhelpful Answer:")
    return text