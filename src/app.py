###
# Elo based comparison of models
# https://chat.lmsys.org/?leaderboard
###

## 
# visual libraries gradio , could be streamlit as well or cl
##
import gradio as gr

##
# Libraries
# Langchain - https://python.langchain.com/docs/get_started/introduction.html
# Used for simplifiing calls, task
##
import langchain
import transformers


# https://huggingface.co/spaces/joyson072/LLm-Langchain/blob/main/app.py
from langchain.llms import HuggingFaceHub

# https://cobusgreyling.medium.com/langchain-creating-large-language-model-llm-applications-via-huggingface-192423883a74
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
#conversation = ConversationChain(
#    llm=llm, 
#    verbose=True, 
#    memory=ConversationBufferMemory()
#)

#conversation.predict(input="Hi there!")


# for the chain and prompt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain 

###################

llm = HuggingFaceHub(
    
    repo_id="google/flan-ul2", 
#    repo_id="google/flan-t5-small", 
    model_kwargs={"temperature":0.1,
                  "max_new_tokens":250})


# Chain 1: Generating a rephrased version of the user's question
template = """{question}\n\n"""
prompt_template = PromptTemplate(input_variables=["question"], template=template)
question_chain = LLMChain(llm=llm, prompt=prompt_template)

# Chain 2: Generating assumptions made in the statement
template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
prompt_template = PromptTemplate(input_variables=["statement"], template=template)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
assumptions_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain], verbose=True
)

# Chain 3: Fact checking the assumptions
template = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
fact_checker_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
)

# Final Chain: Generating the final answer to the user's question based on the facts and assumptions
template = """In light of the above facts, how would you answer the question '{}'""".format(
    "What is the capitol of the usa?"
#    user_question
)
template = """{facts}\n""" + template
prompt_template = PromptTemplate(input_variables=["facts"], template=template)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    verbose=True,
)

#print(overall_chain.run("What is the capitol of the usa?"))

##################



#import model class and tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration


###
# Definition of different purspose prompts
# https://huggingface.co/spaces/Chris4K/rlhf-arena/edit/main/app.py
####
def prompt_human_instruct(system_msg, history):
    return system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["###Human: "+item[0], "###Assistant: "+item[1]])
        for item in history])


def prompt_instruct(system_msg, history):
    return system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["### Instruction: "+item[0], "### Response: "+item[1]])
        for item in history])


def prompt_chat(system_msg, history):
    return system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["USER: "+item[0], "ASSISTANT: "+item[1]])
        for item in history])


def prompt_roleplay(system_msg, history):
    return "<|system|>" + system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["<|user|>"+item[0], "<|model|>"+item[1]])
        for item in history])


####
## Sentinent models
# https://huggingface.co/spaces/CK42/sentiment-model-comparison
# 1, 4 seem best for german
####
model_id_1 = "nlptown/bert-base-multilingual-uncased-sentiment"
model_id_2 = "microsoft/deberta-xlarge-mnli"
model_id_3 = "distilbert-base-uncased-finetuned-sst-2-english"
model_id_4 = "lordtt13/emo-mobilebert"
model_id_5 = "juliensimon/reviews-sentiment-analysis"
model_id_6 = "sbcBI/sentiment_analysis_model"
model_id_7 = "oliverguhr/german-sentiment-bert"

# https://colab.research.google.com/drive/1hrS6_g14EcOD4ezwSGlGX2zxJegX5uNX#scrollTo=NUwUR9U7qkld
#llm_hf_sentiment = HuggingFaceHub(
#    repo_id= model_id_7,
#    model_kwargs={"temperature":0.9 }
#)

from transformers import pipeline

# 
## Possible pipeline
#"['audio-classification', 'automatic-speech-recognition', 'conversational', 'depth-estimation', 'document-question-answering', 
#'feature-extraction', 'fill-mask', 'image-classification', 'image-segmentation', 'image-to-text', 'mask-generation', 'ner', 
#'object-detection', 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification', 
#'text-generation', 'text2text-generation', 'token-classification', 'translation', 'video-classification', 'visual-question-answering', 
#'vqa', 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection', 
#'translation_XX_to_YY']"
##

sentiment_pipe = pipeline("sentiment-analysis", model=model_id_7)
#pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

def pipeline_predict_sentiment(text):
  sentiment_result = sentiment_pipe(text)
  print(sentiment_result)
  return sentiment_result


chat_pipe = pipeline("conversational")

def pipeline_predict_chat(text):
  sentiment_result = chat_pipe(text)
  print(sentiment_result)
  return sentiment_result


#['huggingface', 'models', 'spaces']
#sentiment = gr.load(model_id_7, src="huggingface")

#def sentiment (message):
#  sentiment_label = sentiment.predict(message)
#  print ( sentiment_label)
#  return sentiment_label

#sentiment_prompt = PromptTemplate(
#    input_variables=["text_input"],
#    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
#)

#def sentiment (  message):
#  sentiment_chain = LLMChain(llm=llm_hf_sentiment, prompt=sentiment_prompt)
#  facts = sentiment_chain.run(message)
#  print(facts)
#  return facts





####
## Chat models
# https://huggingface.co/spaces/CK42/sentiment-model-comparison
# 1 seem best for testing
####
chat_model_facebook_blenderbot_400M_distill = "facebook/blenderbot-400M-distill"
chat_model_HenryJJ_vincua_13b = "HenryJJ/vincua-13b"

text = "Why did the chicken cross the road?"

#output_question_1 = llm_hf(text)
#print(output_question_1)



###
## FACT EXTRACTION
###
# https://colab.research.google.com/drive/1hrS6_g14EcOD4ezwSGlGX2zxJegX5uNX#scrollTo=NUwUR9U7qkld
llm_factextract = HuggingFaceHub(
    
#    repo_id="google/flan-ul2", 
    repo_id="google/flan-t5-small", 
    model_kwargs={"temperature":0.1,
                  "max_new_tokens":250})
 
fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
)

def factextraction (message):
  fact_extraction_chain = LLMChain(llm=llm_factextract, prompt=fact_extraction_prompt)
  facts = fact_extraction_chain.run(message)
  print(facts)
  return facts


####
##   models
# 1 seem best for testing
####
#download and setup the model and tokenizer
model_name_chat = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(model_name_chat)
model_chat = BlenderbotForConditionalGeneration.from_pretrained(model_name_chat)

def func (message):
  inputs = tokenizer(message, return_tensors="pt")
  result = model_chat.generate(**inputs)
  print(result)
  return tokenizer.decode(result[0])

title="Conversation Bota"
desc="Some way ... "
app = gr.Interface(
    fn=func,
    title="Conversation Bota",
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)



#####
######
######
examples = [
  ["Erzähl mit eine Geschichte!",50,2,3,1,"Deutsch"],
  ["Welche Blumen sollte man jemandem zum Valentinstag schenken?",50,1,0,1,"Deutsch"],  
  ["Please write a step by step recipe to make bolognese pasta!",50,2,3,2,"Englisch"]
]
tDeEn = pipeline(model="Helsinki-NLP/opus-mt-de-en")
tEnDe = pipeline(model="Helsinki-NLP/opus-mt-en-de")
bot = pipeline(model="google/flan-t5-large")

def solve(text,max_length,length_penalty,no_repeat_ngram_size,num_beams,language):
  if(language=="Deutsch"): 
      text=tDeEn(text)[0]["translation_text"]
  out=bot(text,max_length=max_length, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size, num_beams=num_beams, early_stopping=True)[0]["generated_text"]
  if(language=="Deutsch"): 
      out=tEnDe(out)[0]["translation_text"]
  return out

task = gr.Interface(
  fn=solve,
  inputs=[
      gr.Textbox(lines=5,max_lines=6,label="Frage"),
      gr.Slider(minimum=1.0,maximum=200.0,value=50.0,step=1,interactive=True,label="max_length"),
      gr.Slider(minimum=1.0,maximum=20.0,value=1.0,step=1,interactive=True,label="length_penalty"),
      gr.Slider(minimum=0.0,maximum=5.0,value=3.0,step=1,interactive=True,label="no_repeat_ngram_size"),
      gr.Slider(minimum=1.0,maximum=20.0,value=1.0,step=1,interactive=True,label="num_beams"),
      gr.Dropdown(["Deutsch", "Englisch"],value="Deutsch"),
  ],
  outputs="text",
  title=title,
  description=desc,
  examples=examples
) 


####
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread

model_id = "philschmid/instruct-igel-001"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt_template = f"### Anweisung:\n{{input}}\n\n### Antwort:"

def generate(instruction, temperature=1.0, max_new_tokens=256, top_p=0.9, length_penalty=1.0):
    formatted_instruction = prompt_template.format(input=instruction)
    
    # make sure temperature top_p and length_penalty are floats
    temperature = float(temperature)
    top_p = float(top_p)
    length_penalty = float(length_penalty)
    
    # COMMENT IN FOR NON STREAMING
    # generation_config = GenerationConfig(
    #     do_sample=True,
    #     top_p=top_p,
    #     top_k=0,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    #     early_stopping=True,
    #     length_penalty=length_penalty,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id,
    # )

    # input_ids = tokenizer(
    #     formatted_instruction, return_tensors="pt", truncation=True, max_length=2048
    # ).input_ids.cuda()

    # with torch.inference_mode(), torch.autocast("cuda"):
    #     outputs = model.generate(input_ids=input_ids, generation_config=generation_config)[0]

    # output = tokenizer.decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    # return output.split("### Antwort:\n")[1]

    # STREAMING BASED ON git+https://github.com/gante/transformers.git@streamer_iterator

    # streaming
    streamer = TextIteratorStreamer(tokenizer)
    model_inputs = tokenizer(formatted_instruction, return_tensors="pt", truncation=True, max_length=2048)
    # move to gpu
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generate_kwargs = dict(
        top_p=top_p,
        top_k=0,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        length_penalty=length_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    t = Thread(target=model.generate, kwargs={**dict(model_inputs, streamer=streamer), **generate_kwargs})
    t.start()

    output = ""
    hidden_output = ""
    for new_text in streamer:
        # skip streaming until new text is available
        if len(hidden_output) <= len(formatted_instruction):
            hidden_output += new_text
            continue
        # replace eos token
        if tokenizer.eos_token in new_text:
            new_text = new_text.replace(tokenizer.eos_token, "")
        output += new_text
        yield output
#    if HF_TOKEN:
#        save_inputs_and_outputs(formatted_instruction, output, generate_kwargs)
    return output

#app.launch()
####################


#app_sentiment = gr.Interface(fn=predict , inputs="textbox", outputs="textbox", title="Conversation Bot")
# create a public link, set `share=True` in `launch()
#app_sentiment.launch()
####################

###
###
###
classifier = pipeline("zero-shot-classification")
text = "This is a tutorial about Hugging Face."
candidate_labels = ["informieren", "kaufen", "beschweren", "verkaufen"]

def topic_sale_inform (text):
  res = classifier(text, candidate_labels)
  print (res)
  return res



####
#conversation = Conversation("Welcome")

def callChains(current_message,max_length,length_penalty,no_repeat_ngram_size,num_beams,language):
    #final_answer = generate(current_message,  1.0,  256,  0.9,  1.0)
    sentiment_analysis_result = pipeline_predict_sentiment(current_message)
    topic_sale_inform_result = topic_sale_inform(current_message)
    #conversation.append_response("The Big lebowski.")
    #conversation.add_user_input("Is it good?")
    final_answer = func(current_message)
    #final_answer = solve(current_message,max_length,length_penalty,no_repeat_ngram_size,num_beams,language)
    return final_answer, sentiment_analysis_result, topic_sale_inform_result


###
current_message_inputfield = gr.Textbox(lines=5,max_lines=6,label="Gib hier eine Nachricht ein") 
final_answer_inputfield = gr.Textbox(label="Antwort ", placeholder="Hier kommt die Antwort hin ...")  
sentiment_analysis_result_inputfield = gr.Textbox(label="Sentiment ") 
topic_sale_inform_result_inputfield = gr.Textbox(label="Thema ") 

chat_bot = gr.Interface(fn=callChains , 
                        inputs=[
                                      current_message_inputfield,
                                      gr.Slider(minimum=1.0,maximum=200.0,value=50.0,step=1,interactive=True,label="max_length"),
                                      gr.Slider(minimum=1.0,maximum=20.0,value=1.0,step=1,interactive=True,label="length_penalty"),
                                      gr.Slider(minimum=0.0,maximum=5.0,value=3.0,step=1,interactive=True,label="no_repeat_ngram_size"),
                                      gr.Slider(minimum=1.0,maximum=20.0,value=1.0,step=1,interactive=True,label="num_beams"),
                                      gr.Dropdown(["Deutsch", "Englisch"],value="Deutsch"),
                                  ],
                        outputs=[final_answer_inputfield,sentiment_analysis_result_inputfield,topic_sale_inform_result_inputfield], 
                        title="Conversation Bot with extra")
# create a public link, set `share=True` in `launch()
chat_bot.launch()
####################


app_facts = gr.Interface(fn=factextraction , inputs="textbox", outputs="textbox", title="Conversation Bots")
# create a public link, set `share=True` in `launch()
#app_facts.launch()
####################


