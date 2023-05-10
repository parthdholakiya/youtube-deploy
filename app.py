import streamlit as st
import pickle
import pandas as pd


# Load the necessary data
smd = pd.read_pickle('smd.pkl')
similarity = pd.read_pickle('YouTube_similarity.pkl')
indices_map = pd.read_pickle('indices_map.pkl')
indices = pd.read_pickle('indices.pkl')
svd = pickle.load(open('svd.pkl','rb'))

# Define the hybrid recommender function
def hybrid(title):
    idx = indices[title]
    sim_scores = list(enumerate(similarity[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]
    video_indices = [i[0] for i in sim_scores]
    video = smd.iloc[video_indices][['title','categoryId','video_id','likes','thumbnail_link']]
    video['est'] = video['categoryId'].apply(lambda x: svd.predict(12,indices_map.iloc[x]['video_id']).est)
    video = video.sort_values('likes', ascending=False)
    recomend_video = video[['title','thumbnail_link']].values.tolist()
    return recomend_video

# Set up the app layout
st.set_page_config(page_title='Hybrid YouTube Recommender System', page_icon=':tv:', layout='wide')
st.title('Hybrid YouTube Recommender System')
st.markdown("""
    Discover new videos based on your preferences
    Select a video from the dropdown list and click the "Show Recommendation" button to get started!
""")

# Create the video selection dropdown
video_list = smd['title'].values
selected_video = st.selectbox(
    "Select a video",
    video_list
)

# Create the "Show Recommendation" button and display the recommended videos
if st.button('Show Recommendation'):
    recomend_video = hybrid(selected_video)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            if i < len(recomend_video):
                st.image(recomend_video[i][1], use_column_width=True)
                st.subheader(recomend_video[i][0])
            else:
                break

# Add a footer with some information about the project and its creator
st.markdown("""
    ---
    **About this project:** This project is a hybrid YouTube recommender system that recommends similar videos to the one selected by the user, using a combination of content-based and collaborative filtering techniques. The content-based approach is based on the similarity of video tags, while the collaborative filtering approach is based on user likes. The system uses matrix factorization (SVD) to predict likes and generate recommendations.
    
""")

# Add a sidebar with the YouTube logo and some information about the project creator
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/YouTube_Logo_2017.svg/1200px-YouTube_Logo_2017.svg.png', width=200)
st.sidebar.markdown("""
    **Created by:** Parth N Dholakiya
    This project was created as part of NLP course at Purdue University.
    Contact me at parthdholakiya180@gmail.com for more information.
""")

