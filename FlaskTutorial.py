from flask import Flask, render_template, url_for, request, flash
# from inputForm import inputTeamsForm
#Added render_template function 



app = Flask(__name__) #setting app variable to an instance of flask class (we instantiated a flask application in this app variable) , __name__ is just the name of the module
                      #Its for flask to know where to look for your templates and static files 

@app.route("/",  methods=['GET','POST']) #"/" represents the root page of our website (home page)
def runHtmlPage():
    # form = inputTeamsForm()
    return render_template('FutbolPredictingBlog.html') #Each time we make changes we want to stop the webserver and run again 
                               #When developping it could be a major problem to run the server each time therefore we can run it in "debug mode" which shows the changes while we code 
                               #TO LINK HTML FILES WE WANT TO USE TEMPLATES

# @app.route("/about") #"/about" represents the root page of our website (home page)
# def about():
#     return render_template('about.html')

# @app.route("/", methods=['GET','POST'])
# def inputTeams():
#     form = inputTeamsForm()
#     return render_template('FutbolPredictingBlog.html', title='InputTeams', form=form) 
    # if request.method == 'POST':
    #     HomeTeam = request.form['HomeTeam']
    #     AwayTeam = request.form['AwayTeam']
    #     return render_template('FutbolPredictingBlog.html', HomeTeam = HomeTeam, AwayTeam = AwayTeam)
    
    # return render_template('FutbolPredictingBlog.html')
    # text = request.form['text']
    # if form.validate_on_submit():
    #     flash('Correct Team names!', 'success')
    # processed_text = text.upper()
    # print(processed_text)
    
    #  that way within that template we have access to that form instance in this function

if __name__ == "__main__": #__name__ == __main__ is python if we run the script directly, but if we import this module somewhere else, then the name will be the name of our module  
    app.run(debug=True)     #therefore this conditional is only true if we run this script directly  
                            #now instead of doing flask run we can directly call this script 
###here we can see that the two sections are similar (code from FutbolPredictingBlog.html and about.html), structure is the same just, the body that's different 
###we want each template to only contain its unique information, therefore we'll use template inheritance       






#For base layout we use bootstrap (css and javascript ) add navigation bar/ global styles to our websites / code snippets ...

#To include css file in the layout.html file we are going to use a new flask fuunction called url for (function that finds exactly the routes for us so we don't need to worry about it)

#Flask tutorial : https://www.youtube.com/watch?v=MwZwr5Tvyxo

# create forms and validate input

###IMPORTANT NOTE: FLASK DOES NOT CONSIDER COMMENTS, it reads everything including the comments!!!