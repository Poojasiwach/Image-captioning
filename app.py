from flask import Flask,render_template,redirect,request
import Caption_it 

#__name__ == __main__
app = Flask(__name__) #instanitisng the flask object and passing the module name


#creating routes and handling it
@app.route('/') #by default method is get
def hello():
	return render_template("index.html") 
	#flask knows the html files will placed under the templates folder, it is good to place html files in separate folder

@app.route('/',methods=['POST'])
def caption():
	if request.method == 'POST':
		f = request.files['userfile']
		path = "./static/{}".format(f.filename) #./static/image.jpg
		f.save(path)
		caption = Caption_it.caption_this_img(path)
		
		result_dic = {
		 'image':path,
		 'caption':caption
		}

	return render_template("index.html",your_result=result_dic)



#to run the server
if __name__ == '__main__':
	#app.debug = True
	app.run(debug=True)