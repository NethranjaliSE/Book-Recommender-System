import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_poster(self,suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating =  pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        
        except Exception as e:
            raise AppException(e, sys) from e
        


    def recommend_book(self,book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                    books = book_pivot.index[suggestion[i]]
                    for j in books:
                        books_list.append(j)
            return books_list , poster_url   
        
        except Exception as e:
            raise AppException(e, sys) from e


    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    
    def recommendations_engine(self,selected_books):
        try:
            recommended_books,poster_url = self.recommend_book(selected_books)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_books[1])
                st.image(poster_url[1])
            with col2:
                st.text(recommended_books[2])
                st.image(poster_url[2])

            with col3:
                st.text(recommended_books[3])
                st.image(poster_url[3])
            with col4:
                st.text(recommended_books[4])
                st.image(poster_url[4])
            with col5:
                st.text(recommended_books[5])
                st.image(poster_url[5])
        except Exception as e:
            raise AppException(e, sys) from e



if __name__ == "__main__":

    st.set_page_config(
        page_title="Book Recommender",
        page_icon="📚",
        layout="wide"
    )

    # ---------- Custom CSS ----------
    st.markdown("""
        <style>
        body {
            background-color: #0f172a;
            color: #e5e7eb;
        }

        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 5px;
            animation: fadeIn 1.5s ease-in;
        }

        .sub-title {
            text-align: center;
            color: #9ca3af;
            margin-bottom: 30px;
            animation: fadeIn 2s ease-in;
        }

        .card {
    background: linear-gradient(145deg, #020617, #020617);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: slideUp 0.8s ease;
}

        .card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 20px 40px rgba(0,0,0,0.8);
}
                
                .card-title {
    color: #f8fafc;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
    min-height: 42px;
}

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.title("📚 Book Recommender")
        st.caption("Collaborative Filtering System")
        st.divider()

        if st.button("⚙️ Train Recommender System"):
            obj = Recommendation()
            obj.train_engine()

    # ---------- Main Content ----------
    st.markdown("<div class='main-title'>📖 Book Recommender System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Discover books similar to your favorite one</div>", unsafe_allow_html=True)

    obj = Recommendation()

    # Load book names
    book_names = pickle.load(open(os.path.join('templates','book_names.pkl'), 'rb'))

    st.markdown("### 🔍 Select a Book")
    selected_books = st.selectbox(
        "Type or choose a book",
        book_names
    )

    if st.button("✨ Show Recommendation"):
        st.markdown("### 📚 Recommended Books")

        recommended_books, poster_url = obj.recommend_book(selected_books)

        col1, col2, col3, col4, col5 = st.columns(5)

        for col, book, img in zip(
            [col1, col2, col3, col4, col5],
            recommended_books[1:6],
            poster_url[1:6]
        ):
            with col:
                st.markdown(f"""
    <div class="card">
        <div class="card-title">{book}</div>
    </div>
""", unsafe_allow_html=True)

                st.image(img, width=200)
