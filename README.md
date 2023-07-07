# Movie Recommender System with Tkinter GUI

This is a movie recommender system implemented using Python and Tkinter GUI library. The recommender system suggests movies based on different criteria such as genre and user ratings. The application uses various data preprocessing techniques, including data cleaning, feature extraction, and collaborative filtering, to provide accurate movie recommendations to the users.

## Libraries Used

The following libraries are used in this project:
- pandas: For data manipulation and analysis
- numpy: For numerical operations
- ast: For evaluating strings containing Python code
- nltk.stem.snowball: For stemming words
- sklearn.feature_extraction.text: For converting text to numerical data
- sklearn.metrics.pairwise: For computing cosine similarity
- surprise: For collaborative filtering recommendation algorithms
- collections: For creating defaultdict
- matplotlib.pyplot: For data visualization
- plotly.express: For interactive plots
- wordcloud: For generating word clouds
- seaborn: For statistical data visualization
- networkx: For creating and manipulating graphs

## Data

The project uses the following datasets:
- movies_metadata.csv: Contains movie metadata including genres, vote average, and vote count.
- credits.csv: Contains movie credits data including cast and crew information.
- keywords.csv: Contains keywords associated with movies.
- links_small.csv: Contains movie IDs for linking with other datasets.
- ratings_small.csv: Contains movie ratings provided by users.

## Functionality

The recommender system provides the following features:

### 1. Simple Recommender

The simple recommender suggests top-rated movies irrespective of genres. It calculates the weighted rating for each movie using the IMDB formula and sorts the movies based on the score.

### 2. Genre-based Recommender

The genre-based recommender allows users to select a specific genre and provides recommendations based on that genre. It displays the frequency of occurrence of different genres and recommends the top 10 movies in the selected genre.

### 3. Content-based Recommender

The content-based recommender suggests similar movies based on the content similarity. It creates a word cloud to visualize the frequently occurring words in the dataset. It also creates a cosine similarity matrix to measure the similarity between movies based on their content.

### 4. Collaborative Filtering Recommender

The collaborative filtering recommender uses the Surprise library to implement collaborative filtering recommendation algorithms. It trains the model on the movie ratings dataset and predicts ratings for user-movie pairs. It recommends top movies based on these predictions.

### 5. User Interface with Tkinter

The recommender system is integrated with a graphical user interface (GUI) using the Tkinter library. Users can rate selected movies and choose a genre to get personalized movie recommendations. The recommendations are displayed on the interface.

## How to Run the Application

To run the movie recommender system:

1. Install the required libraries mentioned above.
2. Place the datasets (movies_metadata.csv, credits.csv, keywords.csv, links_small.csv, ratings_small.csv) in the same directory as the Python script.
3. Run the Python script.
4. The GUI window will open, allowing users to rate movies and select a genre.
5. Click the "SUBMIT" button to get the movie recommendations.
6. The recommendations will be displayed on the interface.

## Additional Information

- The movie recommender system uses collaborative filtering techniques to provide personalized recommendations. The accuracy of the recommendations may vary depending on the rating data provided by the users.
- The GUI interface provides an interactive and user-friendly way to interact with the recommender system.
- The code is well-documented with comments to explain the functionality of each section.


