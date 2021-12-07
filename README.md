# Vaccine Sentimental Analysis with Tweets

## Table of Contents

- [Project Description](#project-description)
- [Datasets](#datasets)
- [Sentiment Models](#sentiment-models)
- [Sentiment Analysis Fastai](#sentiment-analysis-fastai)
- [Contributors](#contributors)
## Project Description 

We perform sentiment analysis with a collected COVID-19 vaccines related tweets dataset to analyze the sentiment of the overall public discussion. A comparative analysis with the vaccine progress and the result of sentiment analysis of public discussion on Twitter are also conducted to provide a better understanding of how the public discussion is correlated with vaccine progress. Our task is to perform sentiment analysis on tweets about COVID-19 vaccines by using the dataset of tweets sentiments from all over the world. The datasets include tweets about people’s perspectives towards different kinds of vaccines, the number of different vaccinations per day, week, month and country over the number of people being vaccinated. By performing sentiment analysis, we can get an overall prediction of the relationship between vaccination rates and people’s discussions on Twitter.

## Datasets

> **tweets_covid_all_vaccinations**: all tweets about the COVID-19 vaccines used in entire world and they all include vaccine brands such as Pfizer/BioNTech, Sinopharm, Sinovac, Moderna and etc.

> **tweets_extraction**: the extraction data of tweets includes sentiment label for each of the text and marked each as positive, neutral or negative.

> **tweets_world_vaccination**: data includes daily and total vaccination for COVID-19 in the world such as country, date, number of vaccinations

## Sentiment Models

Import the packages and upload the datasets of **tweets_covid_all_vaccinations** and **tweets_extraction**. Using the function of **remove** to remove things such as URLs, hashtags, emojis and append and merge the two datasets with text and sentiment columns only. We transformed negative, positive, and neutral sentiments to 0, 1, and 2 correspondingly for further modeling. Next, we tokenized the texts and performed lowercase, punctuation removal, small token removal, stop words removal, lemmatization, and stemming to them. The only left columns for tweets include **original text**, **sentiment**, **final text**, and **text tokens**. 

Applied **Naive Bayes** to the dataset and obtained an accuracy of 0.6260 and another baseline model **XGBoost** which gives the test accuracy of 0.696 and trainning accuracy of 0.768 around with multiple XGB models. The third method is deep learning based model CNN to train the sentiment of negative, positive, neutral. Word2vec is a model pre-trained on a large corpus. It provides embeddings that map words that are similar close to each other. A quick way to get a sentence embedding for our classifier is to average word2vec scores of all words. There are totally 21270 unique tokens after vectorization. For the function of ConvNet, the embeddings matrix is passed to embedding_layer. There are five filter sizes applied and GlobalMaxPooling1D layers are applied to each layer. All the outputs are then concatenated. For the **model.summary()**, it will print a brief summary of all the layers with there output shapes. Finally, the number of epochs is the amount to that the model will loop around and learn. The batch size is the amount of data that the model sees at a single time.

## Sentiment Analysis Fastai

Import fastai's text module to handle preprocessing and tokenization. Upload the datasets of **tweets_covid_all_vaccinations** and **tweets_extraction**. Using the function of **remove** to remove things such as URLs, hashtags, emojis and append and merge the two datasets with text and sentiment columns only.

Training the language model using self-supervised learning method. We give the model some texts as independent variables and **fastai** can automatically preprocess it. Using **DataLoaders** class to converts the input to a **dataloader** object. We told **fastai** that we are working with text data, which is contained in the **text** column of a **pandas** **DataFrame** called **df_lm**. We set **is_lm=True** since we want to train a language model, so **fastai** needs to label the input data for us. Finally, we told fastai to hold out a random 10% of our data for a validation set using **valid_pct=0.1**.

### Fine-tunning the language model

Create a language model using **language_model_learner** and pass in the **dataloaders**, **dls_lm**, pre-trained RNN model **AWD_LSTM** and reduce overfitting. As for metrics, the exponential of the loss is perplexity and accuracy to use less memory and accelerate the training process. We get a good learning rate and use it to fit the model. The learning rate has a valley point at almost 0.00302. Using **fit_one_cycle** with our **Learner** to train the new random embeddings in the last layer of the neural network. With one epoch, the language model is predicting the next word in a tweet around 25% of the time. We need to find a more suitable learning rate and train for more epochs to improve the accuracy. With more training, the language model can predict the next word in a tweet around 29% of the time. We can test the model using some written random tweets. Finally we save the model so that it can be used to fine-tune the classifier.

### Fine-tuning the classifier

Using discriminative learning rates and gradual unfreezing each layer until the entire model to train the classifier. This time the model can predict sentiment around 75% of the time. We can do better training with a larger dataset or different model hyperparameters. So much easy to check the model by calling **predict**. The return results are the predicted sentiment with the index of prediction, probabilities for negative, neutral and positive sentiment. 

### Analyze tweets

Adding the tweets to **DataLoaders** as a test set for sentiment analysis. We can plot the bar graph by normalizing it and see the results. 

### Timeline analysis

The function of **filtered_timeline** is to filter the data to a single vaccine and plot the timeline and the function of **date_filter** is to filter the data to a single date and print tweets from users with the most followers. For each of the vaccine including **Sinovac**, **Sinopharm**, **Moderna**, **Sputnik V**, **Pfizer/BioNTech**, **Oxford/AstraZeneca**, we can get the date, original text. There are different increasing rates for each country or region from the results. 


## Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/MadaoIsMyBrother/sentiment-analysis/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MadaoIsMyBrother/sentiment-analysis" />
</a>

Made with [contrib.rocks](https://contrib.rocks).