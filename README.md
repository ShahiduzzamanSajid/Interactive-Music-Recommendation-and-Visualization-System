## Interactive-Music-Recommendation-and-Visualization-System

## Project Overview
The Music Recommendation System project is designed to enhance music exploration and discovery by leveraging data-driven insights and machine-learning techniques. Its primary objective is to provide users with a platform to explore and visualize trends in music data while offering personalized song recommendations based on their preferences. This project is a Music Recommendation System built using machine learning and Streamlit. The application allows users to explore music data through interactive visualizations and generate personalized song recommendations based on user preferences. The system leverages the cosine similarity metric to recommend songs similar to user-provided seed songs.


## Features
 **Data Visualization :**

- Number of Songs Released per Decade: Displays a count of songs released over different decades.
- Popularity Trends Over the Years: Shows how the popularity of music has changed over time.
- Top Genres by Popularity: Highlights the most popular music genres.
- Top Artists by Popularity: Showcases the most popular music artists.
- Top Songs by Popularity: Lists the most popular songs in the dataset.
  
 **Song Recommendations :**

- **Recommend Songs :** Users can input their favorite songs as seed songs, and the system will generate a list of similar songs based on attributes such as valence, acoustics, and danceability.


# Libraries and Tools Used

- **Streamlit :** A framework for building interactive web applications.
- **Pandas :** A data manipulation and analysis library for Python.
- **NumPy :** A library for numerical operations in Python.
- **Seaborn :** A library for creating statistical data visualizations.
- **Plotly :** A library for interactive data visualizations and charts.
- **Matplotlib :** A library for creating static plots.
- **WordCloud :** A library for generating word clouds from text data.
- **Scipy :** A library for scientific computing, including spatial distance calculations.
- **Scikit-Learn :** A machine learning library for Python, used for preprocessing data and calculating similarity.
- **Features Used :**
  - **MinMaxScaler :** Normalizes features to a range between 0 and 1.
  - **StandardScaler :** Standardizes features by removing the mean and scaling to unit variance.
  - **Cosine Similarity :** Measures the cosine of the angle between two non-zero vectors.

## Installation Instructions
To get a local copy of the project up and running, follow these simple steps:

**1. Clone the Repository :**

   ```bash
   git clone https://github.com/ShahiduzzamanSajid/Interactive-Music-Recommendation-and-Visualization-System.git
   cd Interactive-Music-Recommendation-and-Visualization-System
```

**2. Install the Required Libraries :**

To install all the necessary libraries and dependencies for the project, you can use the following commands:

```bash
pip install pandas
pip install numpy
pip install seaborn
pip install plotly
pip install matplotlib
pip install wordcloud
pip install scipy
pip install scikit-learn
pip install imbalanced-learn
pip install streamlit
```

**3. Run the Streamlit App :**

```bash
streamlit run music_recommendations.py
```

## Contributing
Contributions to improve Real-Time Spam ham Detection are welcome. To contribute, follow these steps :

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them with clear comments.
4. Push your changes to your fork.
5. Open a pull request, explaining the changes made.

## License
This project is licensed under the MIT License.

