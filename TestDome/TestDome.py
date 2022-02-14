import spacy

def multi_language(string, model):
    
    NER = spacy.load("en_core_web_sm")
    text1= NER(string)
    
    result = []
    for word in text1.ents:
        #print(word.text,word.label_, word.start, word.end)
        result.append((word.text,word.label_, word.start, word.end))
    
    return result    

if __name__ == "__main__":
    model = "en_core_web_sm"
    string = "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."
    result = multi_language(string, model)
    print(result)