from flask import Flask, render_template, url_for, request
from state import StateTracker

app = Flask(__name__)

#initialize state tracker object 
state_tracker = StateTracker()


@app.route("/",  methods=['GET','POST']) #"/" represents the root page of our website (home page)
def runHtmlPage():
    if request.method == 'POST':
        state_tracker.updateState(request)
        
        return render_template('FutbolPredictingBlog.html', state=state_tracker.state) #define state, that way we don't need to define infinite arguments

    else:
        return render_template('FutbolPredictingBlog.html', state=state_tracker.state) #We have to wait because home team is only assigned when it gets the signal from POST 

if __name__ == "__main__":  
    app.run(debug=True, use_reloader=True)
