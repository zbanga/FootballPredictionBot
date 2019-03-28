from flask import Flask, render_template, url_for, request
from state import StateTracker, preprocess, predict, output, newDF


app = Flask(__name__)

#initialize state tracker object 
state_tracker = StateTracker()

@app.route("/",  methods=['GET','POST']) #"/" represents the root page of our website (home page)
def runHtmlPage():
    if request.method == 'POST':
        state_tracker.updateState(request)
        newDf = newDF(state_tracker)   
        scaledData = preprocess(newDf)
        newDf1 = scaledData.iloc[0,0]
        newDf2 = scaledData.iloc[0,1] 
        prediction = predict(scaledData)            
        whoWins = output(prediction, scaledData)        
        return render_template('FutbolPredictingBlog.html', state=newDf1, state2 = newDf2, prediction = whoWins) #define state, that way we don't need to define infinite arguments

    else:
        return render_template('FutbolPredictingBlog.html', state="", state2 = "", prediction = "") #We have to wait because home team is only assigned when it gets the signal from POST 

if __name__ == "__main__":  
    app.run(debug=True, use_reloader=True)
