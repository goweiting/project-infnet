#!/usr/bin/python

from bottle import get, post, request, route, run
import infnet_main as infnet

import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

@route('/query')
def query():
    return '''
        <form action="/query" method="post">
            Query String: <input name="query_str" type="text" />
            <input value="Search" type="submit" />
        </form>
    '''

@route('/query', method='POST')
def do_query():
    query_str = request.forms.get('query_str')
    chosen_tokens, sorted_uniq_authors = infnet.get_authors_for_query(pub_token_map, authors, authors_alias, publications, query_str)
    if (len(chosen_tokens) == 0 and len(sorted_uniq_authors) == 0):
        return "<h1> We do not know this keyword:"+query_str +"</h1>"
    
    #Build the html.
    html="<p> Query string:"+query_str+"<br><br><b>Related tokens:</b><br>"
    for score, tok in chosen_tokens:
        html+=tok+", "
    html+="<br><br><hr><b>Related Authors:</b><br>"
    for author_name, score in sorted_uniq_authors:
        html+=author_name+"<br>"
    html+="<hr></p>"
    html+='<img src="/images/collab_graph.png" alt="Collaboration Graph">'
    return html


from bottle import static_file
@route('/<filename:path>')
def send_static(filename):
    ''' This makes the extant template start working
       Woo-Hoo!
    '''
    return static_file(filename, root=dir_path+'/'+'static/') 

pub_token_map, authors, authors_alias, publications = infnet.load_files()

run(host='129.215.91.177', port=8080, debug=True)
