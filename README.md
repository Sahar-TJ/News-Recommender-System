# News-Recommender-System

## Introduction
In the current digital age, the volume of information available online is overwhelming, and users often struggle to find relevant content amidst the noise. This challenge is particularly evident in the realm of news consumption, where timely access to pertinent articles is crucial for staying informed. To address this issue, we have developed a sophisticated News Recommender System that leverages advanced Natural Language Processing (NLP) techniques to enhance user experience by providing personalized news recommendations.

The primary objective of this project is to build a robust system that can recommend news articles based on user queries. The system not only identifies relevant articles but also classifies them into predefined categories (World, Sports, Business, Sci/Tech), extracts key information, and summarizes the content. To achieve this, we utilize a combination of traditional NLP methods for preprocessing and state-of-the-art models such as BERT and DistilBART for classification and summarization, respectively. An intuitive web app using Streamlit is also developed to enhance user experience.


__Dataset:__ We utilize the AG News dataset from Kaggle, which offers a diverse collection of news articles. The dataset contains about 127,600 rows and 2 columns which describing the Title and the description of the News. The dataset is further divided equally into 4 different categories. 

__The workflow of our system is as follows:__

- __User Query:__ The user inputs a query, such as "I want to know about Intelligence", "Tell me about the Prime Minister", "Currency", etc.

- __Article Recommendation:__ Using cosine similarity and TF-IDF vectorization, the system retrieves the top five news articles related to the query.

- __Article Classification:__ Each recommended article is classified into one of the predefined categories using a fine-tuned BERT model from Hugging Face.

- __Keyword Extraction:__ For each article, key terms are extracted using traditional NLP techniques to provide a quick overview of the content.

- __Summarization:__ A concise summary of each article is generated using the DistilBART model, allowing users to grasp the essential information quickly.


__Furthermore, we evaluate the performance of our system using various metrics:__

- __Classification Performance:__ Accuracy, precision, recall, and F1-score are used to assess the effectiveness of the article classification.

- __Summarization Performance:__ The BLEU score is utilized to evaluate the quality of the generated summaries against the original article descriptions.
By integrating these advanced NLP techniques, our News Recommender System aims to provide users with an efficient and personalized news consumption experience, ensuring they receive the most relevant and high-quality information.

---

## Steps to Run

### Steps to Run `News_Recommender.ipynb`

__1. Set Up Environment__

- Ensure Python and Jupyter are installed.
- Install Jupyter using pip if necessary:
  ```
  pip install jupyter
  ```
__2. Install Required Libraries__

Install necessary Python libraries
```
pip install pandas numpy matplotlib seaborn nltk sumy transformers scikit-learn torch
```
__3. Download NLTK Data__

Ensure NLTK data files are available by including the following in your notebook
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```
__4. Download the Datasets__

Ensure to download the dataset from [Google Drive Link](https://drive.google.com/drive/folders/1FRe7nxL7w59LAjl9zoiNwVktWUkA1cgB?usp=sharing)
- Datasets required: 
```
train.csv
test.csv
```

__5. Load Jupyter Notebook__

Open terminal or command prompt.

Navigate to the directory containing News_Recommender.ipynb.

Start Jupyter Notebook:
```
jupyter notebook
```

__6. Open and Run Notebook__

In the Jupyter Notebook interface, open News_Recommender.ipynb.

Execute cells sequentially by selecting each cell and pressing Shift + Enter.

_Note: Some code blocks requires the user to enter queries._

---

### Steps to Run App File 'app2.py'

1. Set Up Environment
- Ensure Python is installed on your system.
- Install necessary libraries using pip:

```
  pip install streamlit pandas numpy nltk transformers sklearn torch sumy matplotlib seaborn
```

__2. Prepare Data Files__

Ensure to download the dataset from [Google Drive Link](https://drive.google.com/drive/folders/1FRe7nxL7w59LAjl9zoiNwVktWUkA1cgB?usp=sharing)
- Datasets required: 
```
dataset.csv
```
Ensure that dataset.csv file is in the same directory as app2.py.

__3. Run the Streamlit Application__

- Open terminal or command prompt.

- Navigate to the directory containing app2.py.

- Start the Streamlit application:
```
streamlit run app2.py
```

__4. Access the Application__

After running the command, Streamlit will start a local server.

Open a web browser and go to the URL provided by Streamlit `(usually http://localhost:8501)`

__5. Interact with the Application__

Use the web interface to input the query and get personalized news recommendations.

- Make sure click the buttons 'Recommend' or 'Show Details' to get the results.
- Check out the images for reference

_Note: The user has the freedom to change the length of the summary by using the slider._
