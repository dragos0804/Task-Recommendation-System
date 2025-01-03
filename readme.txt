  _____________________________
 /                             \
|                               |
|	Project structure:      |      
|                               |
 \_____________________________/


	recommendation system +-- data (contains your json file)
			      |
  	                      +-- src +-- model +-- cosine_similarity_recommendation_system.py
  	    		      |       |         |
   	   		      |	      |         +-- __pycache__
    	                      |       |
     	 		      |	      +-- run_me.ipynb  
          	              |
         	              +-- cos.yml (local anaconda environment)
                	      |
               		      +-- readme.txt

  _____________________________
 /                             \
|                               |
|	      Setup:            |      
|                               |
 \_____________________________/


To create an anaconda local environment contains all needed packages, in the conda terminal, type:

	> conda create -n cos python=3 pandas numpy matplotlib seaborn scikit-learn typing

I left the .yml file (environment file) if you'd like to use it instead, to create the environment. 
You may run the following command inside this directory, using the anaconda prompt (conda terminal):

	> conda env create -f cos.yml

If you have troubles with any of these methods, make sure you install matplotlib, seaborn, scikit-learn,
and typing in your local environment or google colab. <3


Run the very suggestively named run_me.ipynb file. It will make use of the cosine similarity algorithm I 
have left under './src/models'. It will also plot some of the data, to make it easier for you to
visualize the results of this system.
