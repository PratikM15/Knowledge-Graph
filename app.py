from flask import Flask, redirect, url_for, request, render_template, send_file
app = Flask(__name__)

import pandas as pd
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
pd.set_option('display.max_colwidth', 200)

def get_entities(sent):
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""   
    prv_tok_text = ""
    prefix = ""
    modifier = ""
    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text
        if tok.dep_.endswith("mod") == True:
            modifier = tok.text
            if prv_tok_dep == "compound":
                modifier = prv_tok_text + " "+ tok.text
        if tok.dep_.find("subj") == True:
            ent1 = modifier +" "+ prefix + " "+ tok.text
            prefix = ""
            modifier = ""
            prv_tok_dep = ""
            prv_tok_text = ""
        if tok.dep_.find("obj") == True:
            ent2 = modifier +" "+ prefix +" "+ tok.text
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    return [ent1.strip(), ent2.strip()]

def get_relation(sent):
    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)
    pattern = [[{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}]]
    matcher.add("matching_1", pattern)
    matches = matcher(doc)
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]]
    return(span.text)

@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        sentences = text.split('.')
        sentences = [sentence for sentence in sentences if sentence!='']
        entity_pairs = []
        for i in tqdm(sentences):
            entity_pairs.append(get_entities(i))
        try:
            relations = [get_relation(i) for i in tqdm(sentences)]
        except:
            return render_template('index.html', msg="Try with another text.")
        source = [i[0] for i in entity_pairs]
        target = [i[1] for i in entity_pairs]
        kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
        graph = []
        for i in range(len(source)):
            graph.append([source[i], relations[i], target[i]])
        kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
        kg_df.to_csv('./files/recent.csv')
        G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
        plt.figure(figsize=(12,12))

        pos = nx.spring_layout(G)
        nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
        #plt.show(block=False)
        plt.savefig("./files/Graph.png", format="PNG")
        return render_template('index.html', graph=graph, text=text)
    else:
        return render_template('index.html')

@app.route('/files/<path:filename>')
def downloadFile(filename):
    path = "files/"+filename
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug = True)
