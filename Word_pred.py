import os
import streamlit as st
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM

import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(page_title='Next Word Prediction Model', page_icon=None, layout='centered', initial_sidebar_state='auto')


names = ['Akhil Mathew','Neenu Markose']
usernames = ['amathew','nmarkose']
passwords = ['123','456']

hashed_passwords = stauth.hasher(passwords).generate()

authenticator = stauth.authenticate(names,usernames,hashed_passwords,'cookie_name', 'signature_key',cookie_expiry_days=30)

name, authentication_status = authenticator.login('Logout','main')

if authentication_status:
 st.sidebar.write('Welcome *%s*' % (name))
 # your application
 
 @st.cache()
 def load_model(model_name):
  try:
    if model_name.lower() == "bert":
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer,bert_model
  except Exception as e:
    pass

#use joblib to fast your function

 def decode(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

 def encode(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx

 def get_predictied_word(res, top_clean=5):
    # ========================= BERT =================================
    i=0
    c = 0
    list_words=[]
    a = res[0]
    output_text=''
    if a!="":
        for w in res:
            if w!='<mask>':
                input_text = a +' <mask>'
                input_ids, mask_idx = encode(bert_tokenizer, input_text)
                with torch.no_grad():
                    predict = bert_model(input_ids)[0]
                bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
                if (i+1) < len(res):
                    output_text = res[i+1]
                    if output_text in bert:
                        list_words.append(output_text)
                        #print(list_words)
                        c+=1
                    
            a = a+' '+output_text 
            #print(a)
            i = i+1
        
    len_predicted_words = len(list_words)   
    return c,'\n'.join(list_words[:len_predicted_words])

 def get_all_predictions(text_sentence, top_clean=10):
    # ========================= BERT =================================
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
  word_predicted=''
  accuracy=0
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  res = len(text_sentence.split())
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
  c,word_predicted = get_predictied_word(text_sentence.split())
  if (res-2)>0: 
      accuracy = (c/(res-2)) * 100
  return {'bert': bert,'input_length':res-1, 'predicted_words_length':c,'predictied_words_used':word_predicted,'accuracy':accuracy}

 def get_prediction_eos(input_text):
  try:
    input_text += ' <mask>'
    res = get_all_predictions(input_text, top_clean=int(top_k))
    return res
  except Exception as error:
    pass

 try:

  st.markdown("<h1 style='text-align: center;'>Next Word Prediction</h1>", unsafe_allow_html=True)

  top_k = st.sidebar.slider("Select How many words do you need", 1 , 10, 3) #some times it is possible to have less words
  print(top_k)
  #model_name = st.sidebar.selectbox(label='Select Model to Apply',  options=['BERT', 'XLNET'], index=0,  key = "model_name")

  bert_tokenizer, bert_model  = load_model('BERT') 
  input_text = st.text_area("Enter your text here")

  #click outside box of input text to get result
  res = get_prediction_eos(input_text)

  answer = []
  acc = []
  print(res['bert'].split("\n"))
  for i in res['bert'].split("\n"):
  	answer.append(i)
  answer_as_string = "    ".join(answer)


  if st.button('Predict'):
      st.text_area("Predicted List is Here",answer_as_string,key="predicted_list")
  st.image('https://imageio.forbes.com/blogs-images/cognitiveworld/files/2019/06/types-of-AI.jpg?format=jpg&width=960',use_column_width=True)

 except Exception as e:
  print("SOME PROBLEM OCCURED")
  
elif authentication_status == False:
 st.error('Username/password is incorrect')
elif authentication_status == None:
 st.warning('Please enter your username and password')