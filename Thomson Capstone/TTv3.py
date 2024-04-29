import streamlit as st
import pandas as pd
import string
import re
import nltk
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Dictionary to map food choices to their descriptions
food_choices = {
    "Hainanese Chicken Rice": "Hainanese chicken rice is a beloved dish consisting of tender poached chicken served with flavorful rice, accompanied by chili sauce and ginger paste. The chicken is typically succulent and infused with subtle aromas, while the rice is rich in flavor from being cooked in chicken broth.",
    "Laksa": "Laksa is a spicy noodle soup with a rich and creamy coconut curry broth. It's typically served with rice noodles and topped with shrimp, fish cakes, bean sprouts, and a hard-boiled egg. The broth is flavorful and aromatic, with a perfect balance of spice and creaminess.",
    "Char Kway Teow": "Char Kway Teow is a popular stir-fried noodle dish known for its smoky flavor and savory sauce. It features flat rice noodles cooked with Chinese sausage, shrimp, cockles, bean sprouts, and chives, all stir-fried in a flavorful dark soy sauce-based seasoning.",
    "Chilli Crab": "Chilli Crab is a quintessential Singaporean seafood dish featuring mud crab cooked in a tangy and spicy chili sauce. The sauce is rich and flavorful, with hints of sweetness from the tomato paste and a kick of heat from the chili. It's often enjoyed with mantou (fried buns) for dipping.",
    "Satay": "Satay is a popular street food in Singapore consisting of skewered and grilled meat served with a flavorful peanut sauce. The meat is marinated in a blend of spices, then grilled to perfection, resulting in tender and juicy skewers. It's often served with cucumber slices and rice cakes.",
    "Roti Prata": "Roti Prata is a crispy and flaky flatbread that originated from Indian cuisine but has become a favorite in Singapore. It's typically served with a side of savory curry dipping sauce, which complements the buttery and crispy texture of the prata.",
    "Nasi Lemak": "Nasi Lemak is a traditional Malay dish known for its fragrant coconut rice served with an array of flavorful accompaniments. It's often accompanied by fried chicken or fish, sambal chili for heat, fried anchovies and roasted peanuts for crunch, cucumber slices for freshness, and a hard-boiled egg for protein.",
    "Mee Goreng": "Mee Goreng is a flavorful and spicy stir-fried noodle dish with origins in Indian and Malay cuisine. It features yellow noodles stir-fried with a variety of vegetables, tofu, shrimp, and a spicy and tangy sauce made from a blend of chili, tomato, and spices.",
    "Hokkien Mee": "Hokkien Mee is a popular noodle dish in Singapore made from a mix of yellow and rice noodles stir-fried with seafood, such as shrimp and squid, and vegetables like cabbage and bean sprouts. The dish is infused with a rich seafood flavor from the broth, resulting in a hearty and satisfying meal.",
    "Mee Siam": "Mee Siam is a flavorful noodle dish with Malay and Thai influences, popular in Singapore. It features rice vermicelli noodles cooked in a spicy, sweet, and tangy gravy, served with prawns, tofu, hard-boiled egg, and bean sprouts. It's a vibrant and satisfying dish with a perfect balance of flavors.",
    "Kway Chap": "Kway Chap is a comforting noodle soup dish of Teochew origin popular in Singapore. It features wide rice noodles served in a rich and aromatic broth made from pork bones and various spices, accompanied by tender braised pork, tofu, hard-boiled eggs, and offal. It's a hearty and flavorful meal that's perfect for any time of the day.",
    "Tau Huay": "Tau Huay, also known as tofu pudding or douhua, is a comforting dessert popular in Singapore. It's made from soft and silky smooth tofu pudding served with a sweet syrup, often flavored with pandan leaves or ginger for added fragrance. It's a light and refreshing dessert perfect for a hot day.",
    "Chwee Kueh": "Chwee Kueh is a classic Teochew snack in Singapore made from steamed rice cakes topped with a savory mixture of preserved radish (chye poh) and fried shallots. It's typically served with a side of spicy chili sauce for added flavor and heat. The soft and slightly chewy texture of the rice cakes contrasts beautifully with the crunchy toppings.",
    "Putu Piring": "Putu Piring is a traditional Malay dessert in Singapore made from steamed rice flour cakes filled with gooey gula melaka (palm sugar) and served with freshly grated coconut. It's a sweet and fragrant treat with a melt-in-your-mouth texture, perfect for satisfying your sweet tooth.",
    "Ice Kachang": "Ice Kachang is a beloved dessert in Singapore made from shaved ice topped with colorful syrup, sweetened condensed milk, red beans, attap chee (palm seed), grass jelly, and corn. It's a refreshing and sweet treat that's perfect for cooling down on a hot day.",
    "Mee Rebus": "Mee Rebus is a popular Malay noodle dish in Singapore, featuring yellow noodles drenched in a thick and flavorful sweet potato-based gravy. The gravy is cooked with a blend of spices, dried shrimp, and peanuts, giving it a rich and savory taste. It's typically garnished with boiled egg, fried tofu, bean sprouts, and crispy shallots.",
    "Prawn Noodles": "Prawn Noodles, also known as 'Hae Mee' in Hokkien, is a flavorful noodle soup dish in Singapore made with a rich broth flavored with prawn heads and shells. The broth is simmered for hours to extract maximum flavor and then served with yellow noodles, prawns, slices of pork, hard-boiled egg, and bean sprouts. It's a comforting and hearty dish enjoyed by many.",
    "Kacang Pool": "Kacang Pool is a hearty and savory bean stew dish popular in Singapore's Arab community. It's made from mashed fava beans cooked with tomatoes, onions, garlic, and a blend of spices such as cumin and paprika. It's typically served with toasted bread or pita, making it a satisfying meal for breakfast or brunch.",
    "Nasi Goreng": "Nasi Goreng, which translates to 'fried rice' in Indonesian, is a flavorful fried rice dish popular in Singapore. It's cooked with a mix of spices, garlic, shallots, shrimp paste, and kecap manis (sweet soy sauce), giving it a rich and aromatic flavor. It's often served with fried eggs, sliced cucumbers, and krupuk (shrimp crackers) for added crunch.",
    "Lontong": "Lontong is a traditional Indonesian dish popular in Singapore, consisting of compressed rice cakes served with a coconut milk-based soup and a variety of side dishes such as tofu, hard-boiled eggs, vegetables, and sambal (spicy chili paste). The soup is flavored with spices such as lemongrass, galangal, and turmeric, giving it a fragrant and savory taste.",
    "Mee Soto": "Mee Soto is a comforting noodle soup dish popular in Singapore and Malaysia, made with yellow noodles served in a fragrant and flavorful chicken broth. The broth is cooked with spices such as lemongrass, galangal, and turmeric, giving it a rich and aromatic flavor. It's typically garnished with shredded chicken, bean sprouts, fried shallots, and fresh cilantro, making it a satisfying and hearty meal.",
    "Bubur Cha Cha": "Bubur Cha Cha is a traditional Nyonya dessert popular in Singapore, made from a coconut milk-based soup with sweet potatoes, yam, taro, and tapioca pearls. The soup is flavored with pandan leaves and palm sugar, giving it a fragrant and sweet taste. It's a comforting and delicious dessert enjoyed by many.",
    "Ondeh Ondeh": "Ondeh Ondeh is a popular Malay dessert in Singapore, consisting of glutinous rice balls filled with gula melaka (palm sugar) and coated in grated coconut. The rice balls are boiled until they float to the surface, indicating that they are cooked, and then rolled in grated coconut for added texture. They are sweet and chewy, with a burst of caramelized palm sugar in the center, making them a delightful treat.",
    "Nasi Kerabu": "Nasi Kerabu is a traditional Malay dish popular in Singapore, consisting of blue-colored rice served with an assortment of herbs, vegetables, and proteins. The rice gets its vibrant color from the petals of the butterfly pea flower, and it's typically served with items such as fried chicken, salted egg, fish crackers, and a variety of pickles. It's a flavorful and colorful dish that's as visually appealing as it is delicious.",
    "Ketupat": "Ketupat is a traditional Malay rice cake popular in Singapore, made from woven palm leaves filled with rice and then boiled until cooked. It's often served with dishes such as rendang or satay, and it's a symbol of celebration and festivity, commonly enjoyed during festive occasions such as Hari Raya Aidilfitri. The rice cakes have a firm texture and a subtle flavor, making them a perfect accompaniment to savory dishes.",
    "Kueh Dadar": "Kueh Dadar is a traditional Nyonya dessert popular in Singapore, made from pandan-flavored crepes filled with sweetened grated coconut. The crepes are cooked until they are slightly crispy on the edges and then rolled up with the coconut filling inside. They are fragrant and sweet, with a hint of pandan flavor, making them a delightful treat for any occasion."
}
# Load the data
def load_data():
    data = pd.read_csv("TT_whisky_v3.csv")
    return data

# Preprocess the data
def preprocess_data(data):
    data = data[['BottleName', 'review']]
    data['review_clean'] = data['review'].apply(lambda x: remove_punct(x))
    data['review_tokenized'] = data['review_clean'].apply(lambda x: tokenize(x.lower()))
    data['review_nostop'] = data['review_tokenized'].apply(lambda x: remove_stopwords(x))
    data['review_lemmatized'] = data['review_nostop'].apply(lambda x: lemmatizing(x))
    return data

# Remove punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

# Tokenize text
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

# Remove stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

# Lemmatize text
def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

# Train Word2Vec model
def train_word2vec_model(data):
    w2v_model = Word2Vec(data["review_lemmatized"], vector_size=100, window=30, min_count=5)
    return w2v_model

# Get sentence vector
def get_sentence_vector(sentence):
    word_vectors = get_word_vectors(sentence)
    if not word_vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(word_vectors, axis=0)

# Get word vectors
def get_word_vectors(sentence):
    word_vectors = []
    for word in sentence.split():
        if word in w2v_model.wv:
            word_vectors.append(w2v_model.wv[word])
    return word_vectors

# Find closest sentence
def find_closest_sentence(input_text, data):
    input_vector = get_sentence_vector(input_text)
    max_similarity = -1
    closest_sentence = None
    closest_similarity_percent = 0
    
    for index, row in data.iterrows():
        sentence_vector = row['sentence_vector']
        similarity = cosine_similarity([input_vector], [sentence_vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            closest_sentence = row['BottleName']
            closest_similarity_percent = similarity * 100  # Save the similarity percentage
    
    return closest_sentence, closest_similarity_percent

# Load NLTK stopwords
nltk.download('stopwords')
nltk.download('wordnet')
stopword = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()

# Streamlit app
def main():
    st.title("Whisky Savant")

    # Load data
    data = load_data()
    data = preprocess_data(data)

    # Train Word2Vec model
    global w2v_model
    w2v_model = train_word2vec_model(data)

    # Calculate sentence vectors and store them in the DataFrame
    data['sentence_vector'] = data['review_lemmatized'].apply(lambda x: get_sentence_vector(' '.join(x)))

    # Display food choices and get user input
    selected_foods = select_food_choices()

    # Find the closest review for each selected food
    closest_reviews = []

    for food in selected_foods:
        food_description = food_choices[food]
        closest_sentence, closest_similarity_percent = find_closest_sentence(food_description, data)
        closest_reviews.append((food, closest_sentence, closest_similarity_percent))

    # Sort the closest reviews based on similarity percentage in descending order
    closest_reviews.sort(key=lambda x: x[2], reverse=True)

    # Display the top 5 closest reviews
    st.subheader("Your selection of comfort foods has inspired a personalized whisky journey just for you.")
    for i, (food, closest_sentence, closest_similarity_percent) in enumerate(closest_reviews[:5], start=1):
        st.write(f"{i}. {food}")

    # Combine descriptions of selected foods
    combined_description = " ".join([food_choices[food] for food in selected_foods])

    # Find closest reviews based on combined description
    closest_reviews_combined = []

    for index, row in data.iterrows():
        sentence_vector = row['sentence_vector']
        similarity = cosine_similarity([get_sentence_vector(combined_description)], [sentence_vector])[0][0]
        closest_reviews_combined.append((row['BottleName'], similarity))

    # Sort the closest reviews based on similarity percentage in descending order
    closest_reviews_combined.sort(key=lambda x: x[1], reverse=True)

    # Display the top 5 closest reviews based on the combined description
    st.subheader("Indulge in Our Top 5 Whisky Recommendations Inspired by Your Culinary Preferences:")
    for i, (title, similarity) in enumerate(closest_reviews_combined[:5], start=1):
        st.write(f"{i}. Whisky Name: {title}, Similarity: {similarity * 100:.2f}%")




# Function to select food choices using checkboxes
def select_food_choices():
    st.sidebar.title("Let me know 5 of your comfort food")
    selected_foods = []
    num_selected = 0
    
    # Create two columns layout
    col1, col2 = st.sidebar.columns(2)
    
    # Split food choices into two halves
    total_choices = len(food_choices)
    midpoint = total_choices // 2
    first_half = list(food_choices.items())[:midpoint]
    second_half = list(food_choices.items())[midpoint:]
    
    # Loop through first half and display checkboxes in first column
    for food, description in first_half:
        if num_selected < 5:
            if col1.checkbox(food, key=food):
                selected_foods.append(food)
                num_selected += 1
        else:
            st.sidebar.error("You can only select up to 5 food choices.")
            break
    
    # Loop through second half and display checkboxes in second column
    for food, description in second_half:
        if num_selected < 5:
            if col2.checkbox(food, key=food):
                selected_foods.append(food)
                num_selected += 1
        else:
            st.sidebar.error("You can only select up to 5 food choices.")
            break
    
    return selected_foods

# Add image to the app
st.image("C:/Users/Thomson Tan/OneDrive - Nanyang Technological University/TFIPGA/Thomson Capstone/TT_background.jpg", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()

git add .
git commit -m "Initial commit"
git remote add origin https://github.com/2397806N/WhiskySavant.git
git push -u origin main

git push origin main
