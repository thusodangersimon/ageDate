from flask import Flask, request
from flask.ext.restful import Resource, Api
from glob import glob

app = Flask(__name__)
api = Api(app)

gal_list = {}
#for gal in 

class GalDistribute(Resource):
    '''Stores and distributes names and spectra of galaxies that need to be
    processed'''
    
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def put(self, todo_id):
        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}

api.add_resource(GalDistribute, '/<string:todo_id>')

if __name__ == '__main__':
    app.run(debug=True)
