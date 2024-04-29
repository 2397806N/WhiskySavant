import streamlit as st
import pandas as pd
import string
import re
import nltk
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the food choices and descriptions
food_choices = {
    "Hainanese Chicken Rice": "Hainanese chicken rice is a beloved dish consisting of tender poached chicken served with flavorful rice, accompanied by chili sauce and ginger paste. The chicken is typically succulent and infused with subtle aromas, while the rice is rich in flavor from being cooked in chicken broth.",
    "Laksa": "Laksa is a spicy noodle soup with a rich and creamy coconut curry broth. It's typically served with rice noodles and topped with shrimp, fish cakes, bean sprouts, and a hard-boiled egg. The broth is flavorful and aromatic, with a perfect balance of spice and creaminess.",
    "Char Kway Teow": "Char Kway Teow is a popular stir-fried noodle dish known for its smoky flavor and savory sauce. It features flat rice noodles cooked with Chinese sausage, shrimp, cockles, bean sprouts, and chives, all stir-fried in a flavorful dark soy sauce-based seasoning.",
    "Chilli Crab": "Chilli Crab is a quintessential Singaporean seafood dish featuring mud crab cooked in a tangy and spicy chili sauce. The sauce is rich and flavorful, with hints of sweetness from the tomato paste and a kick of heat from the chili. It's often enjoyed with mantou (fried buns) for dipping.",
    "Satay": "Satay is a popular street food in Singapore consisting of skewered and grilled meat served with a flavorful peanut sauce. The meat is marinated in a blend of spices, then grilled to perfection, resulting in tender and juicy skewers. It's often served with cucumber slices and rice cakes.",
    "Roti Prata": "Roti Prata is a crispy and flaky flatbread that originated from Indian cuisine but has become a favorite in Singapore. It's typically served with a side of savory curry dipping sauce, which complements the buttery and crispy texture of the prata.",
    "Nasi Lemak": "Nasi Lemak is a traditional Malay dish known for its fragrant coconut rice served with an array of flavorful accompaniments. It's often accompanied by fried chicken or fish, sambal chili for heat, fried anchovies and roasted peanuts for crunch, cucumber slices for freshness, and a hard-boiled egg for protein.",
    "Mee Goreng": "Mee Goreng is a flavorful and spicy stir-fried noodle dish with origins in Indian and Malay cuisine. It features yellow noodles stir-fried with a variety of vegetables, tofu, shrimp, and a spicy and tangy sauce made from a blend of chili, tomato, and spices.",
    "Bak Kut Teh": "Bak Kut Teh, which translates to \"meat bone tea,\" is a comforting soup dish consisting of tender pork ribs simmered in a flavorful broth of Chinese herbs and spices. The broth is rich and aromatic, with layers of flavor from ingredients like garlic, pepper, and star anise.",
    "Hokkien Mee": "Hokkien Mee is a popular noodle dish in Singapore made from a mix of yellow and rice noodles stir-fried with seafood, such as shrimp and squid, and vegetables like cabbage and bean sprouts. The dish is infused with a rich seafood flavor from the broth, resulting in a hearty and satisfying meal.",
    "Bak Chor Mee": "Bak Chor Mee is a flavorful noodle dish featuring springy egg noodles tossed in a spicy vinegar sauce and topped with minced pork, pork slices, mushrooms, and meatballs. The combination of savory, tangy, and spicy flavors makes it a favorite among noodle lovers.",
    "Murtabak": "Murtabak is a savory pancake of Arab origin that has become popular in Singapore. It's made from thin, flaky pastry filled with a mixture of minced meat, onions, and egg, then folded and grilled to perfection. It's often served with a side of curry dipping sauce for added flavor.",
    "Kaya Toast": "Kaya Toast is a classic Singaporean breakfast dish consisting of toasted bread spread with kaya, a sweet and creamy coconut and egg jam, and butter. It's typically served with soft-boiled eggs and a cup of coffee or tea, making it a comforting and delicious start to the day.",
    "Rojak": "Rojak is a traditional fruit and vegetable salad dish popular in Singapore and Malaysia. It features a mix of crunchy fruits and vegetables tossed in a sweet and spicy shrimp paste sauce, then topped with crushed peanuts and sesame seeds for added texture and flavor.",
    "Nasi Padang": "Nasi Padang is an Indonesian rice dish that has become popular in Singapore. It consists of steamed rice served with a variety of flavorful and spicy dishes, such as rendang, sambal goreng, and sayur lodeh. It's a feast for the senses, with a mix of bold flavors and aromatic spices.",
    "Biryani": "Biryani is a fragrant and flavorful rice dish of South Asian origin that has become popular in Singapore. It features long-grain basmati rice cooked with a blend of aromatic spices, layered with tender meat (chicken, mutton, or beef), and served with a side of cooling yogurt-based raita.",
    "Otah-Otah": "Otah-Otah is a traditional Malay dish made from spicy fish paste mixed with herbs and spices, then wrapped in banana leaves and grilled or steamed until cooked through. It's a flavorful and aromatic dish with a perfect balance of spice and sweetness from the banana leaves.",
    "Kueh Pie Tee": "Kueh Pie Tee is a popular Peranakan snack in Singapore made from crispy pastry cups filled with a savory mixture of shredded turnips, carrots, and prawns, then topped with chili sauce and fried shallots. It's a delightful combination of textures and flavors, with a hint of sweetness from the turnips.",
    "Tau Huay": "Tau Huay, also known as tofu pudding or douhua, is a comforting dessert popular in Singapore. It's made from soft and silky smooth tofu pudding served with a sweet syrup, often flavored with pandan leaves or ginger for added fragrance. It's a light and refreshing dessert perfect for a hot day.",
    "Chwee Kueh": "Chwee Kueh is a classic Teochew snack in Singapore made from steamed rice cakes topped with a savory mixture of preserved radish (chye poh) and fried shallots. It's typically served with a side of spicy chili sauce for added flavor and heat. The soft and slightly chewy texture of the rice cakes contrasts beautifully with the crunchy toppings.",
    "Mee Siam": "Mee Siam is a flavorful noodle dish with Malay and Thai influences, popular in Singapore. It features rice vermicelli noodles cooked in a spicy, sweet, and tangy gravy, served with prawns, tofu, hard-boiled egg, and bean sprouts. It's a vibrant and satisfying dish with a perfect balance of flavors.",
    "Kway Chap": "Kway Chap is a comforting noodle soup dish of Teochew origin popular in Singapore. It features wide rice noodles served in a rich and aromatic broth made from pork bones and various spices, accompanied by tender braised pork, tofu, hard-boiled eggs, and offal. It's a hearty and flavorful meal that's perfect for any time of the day.",
    "Putu Piring": "Putu Piring is a traditional Malay dessert in Singapore made from steamed rice flour cakes filled with gooey gula melaka (palm sugar) and served with freshly grated coconut. It's a sweet and fragrant treat with a melt-in-your-mouth texture, perfect for satisfying your sweet tooth.",
    "Bak Kwa": "Bak Kwa is a popular Chinese New Year snack in Singapore made from grilled slices of sweet and savory dried meat, typically pork or beef. The meat is thinly sliced and marinated in a blend of spices and sugar before being grilled to perfection, resulting in a tender and flavorful snack that's irresistible.",
    "Ice Kachang": "Ice Kachang is a beloved dessert in Singapore made from shaved ice topped with colorful syrup, sweetened condensed milk, red beans, attap chee (palm seed), grass jelly, and corn. It's a refreshing and sweet treat that's perfect for cooling down on a hot day."
}

# Function to remove punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

# Function to tokenize text
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

# Function to remove stopwords
def remove_stopwords(tokenized_list):
    stopword = nltk.corpus.stopwords.words('english')  # Define stopwords here
    text = [word for word in tokenized_list if word not in stopword]
    return text

# Function to lemmatize text
def lemmatizing(tokenized_text):
    wn = nltk.WordNetLemmatizer()  # Initialize WordNet lemmatizer
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

# Function to get word vectors
def get_word_vectors(sentence, w2v_model):
    word_vectors = []
    for word in sentence.split():
        if word in w2v_model.wv:
            word_vectors.append(w2v_model.wv[word])
    return word_vectors

# Function to calculate sentence vector by averaging word vectors
def get_sentence_vector(sentence, w2v_model):
    word_vectors = get_word_vectors(sentence, w2v_model)
    if not word_vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(word_vectors, axis=0)

# Function to find the closest sentence to the input text and calculate the cosine similarity percentage
def find_closest_sentence(input_text, data, w2v_model):
    input_vector = get_sentence_vector(input_text, w2v_model)
    max_similarity = -1
    closest_sentence = None
    closest_similarity_percent = 0
    
    for index, row in data.iterrows():
        sentence_vector = row['sentence_vector']
        similarity = cosine_similarity([input_vector], [sentence_vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            closest_sentence = row['Title']
            closest_similarity_percent = similarity  # Save the similarity percentage

    return closest_sentence, closest_similarity_percent

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("whisky_clean.csv")
    return data

def main():
    st.title("Whisky Review Recommender")

    # Load data
    data = load_data()

    # Preprocessing
    data = data[['Title', 'Nose', 'Taste', 'Finish']]
    data['combined_text'] = data['Nose'].astype(str) + " " + data['Taste'].astype(str) + " " + data['Finish'].astype(str)
    data['combined_text_clean'] = data['combined_text'].apply(lambda x: remove_punct(x))
    data['combined_text_tokenized'] = data['combined_text_clean'].apply(lambda x: tokenize(x.lower()))

    # NLTK stopwords
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Train Word2Vec model
    w2v_model = Word2Vec(data["combined_text_tokenized"], vector_size=100, window=5, min_count=2)
    review_vect_list = []
    for index, row in data.iterrows():
        model_vector = (np.mean([w2v_model.wv.get_vector(token) for token in row['combined_text_tokenized'] if token in w2v_model.wv], axis=0)).tolist()
        if type(model_vector) is list:  
            review_vect_list.append(model_vector)
        else:
            review_vect_list.append([str(0) for i in range(100)])
    word2vec_df = pd.DataFrame(review_vect_list)
    data['sentence_vector'] = word2vec_df.values.tolist()

# Get food choices
    selected_foods = []
    st.write("Select 3 food choices:")
    for i in range(3):
        food_choice = st.selectbox(f"Food Choice {i+1}", options=list(food_choices.keys()))
        selected_foods.append(food_choice)


    # Find closest review for each selected food
    for food in selected_foods:
        food_description = food_choices[food]
        closest_sentence, closest_similarity_percent = find_closest_sentence(food_description, data, w2v_model)
        st.write(f"For {food}:")
        st.write("Closest Review:", closest_sentence)
        st.write("Cosine Similarity Percentage:", f"{closest_similarity_percent:.2%}")

    # Combine food descriptions
    combined_description = " ".join([food_choices[food] for food in selected_foods])

    # Find closest review based on combined description
    max_similarity_percent_combined = -1
    closest_review = None
    for index, row in data.iterrows():
        sentence_vector = row['sentence_vector']
        similarity = cosine_similarity([get_sentence_vector(combined_description, w2v_model)], [sentence_vector])[0][0]
        if similarity > max_similarity_percent_combined:
            max_similarity_percent_combined = similarity
            closest_review = row['Title']

    # Display closest review based on combined description
    st.write("\nClosest Review based on selected foods:")
    st.write("Review:", closest_review)
    st.write("Cosine Similarity Percentage:", f"{max_similarity_percent_combined:.2%}")

if __name__ == "__main__":
    main()
