import json
import nltk
import os
import regex
import sys
from nltk.stem.porter import PorterStemmer

sys.path.insert(0,f"{os.getcwd()}/acl-anthology/bin")
from anthology import Anthology

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

class Stemmer:
    """
    Provides methods for stemming text and keywords.
    """
    def __init__(self):
        """
        Initializes the PorterStemmer.
        """
        self.stemmer = PorterStemmer()

    def stem_text(self, text):
        """
        Stems the given text, tokenizing it and applying stemming to each word.

        Args:
            text (str): The input text to stem.

        Returns:
            list[str]: A list of stemmed words.
        """
        words = nltk.word_tokenize(text)
        return [self.stemmer.stem(word) for word in words]

    def stem_keywords(self, keywords):
        """
        Splits a string of keywords separated by "|" and stems each keyword.

        Args:
            keywords (str): A string containing keywords separated by "|".

        Returns:
            str: A string of stemmed keywords separated by "|".
        """
        keyword_list = keywords.split("|")
        stemmed_keywords = [self.stem_text(word)[0] for word in keyword_list]
        return '|'.join(stemmed_keywords)


class PaperSelector:
    """
    Filters papers based on stemmed keywords in titles or abstracts.
    """
    def __init__(self, stemmer, anthology):
        """
        Initializes the PaperSelector with a stemmer and anthology.

        Args:
            stemmer (Stemmer): An instance of the Stemmer class.
            anthology (Anthology): An instance of the Anthology class.
        """
        self.stemmer = stemmer
        self.anthology = anthology
        self.selected_papers = {}

    def search_regex(self, pattern, string):
        """
        Searches for the given pattern in a string using regular expressions.

        Args:
            pattern (str): The regex pattern to search for.
            string (str): The string to search within.

        Returns:
            bool: True if the pattern is found, False otherwise.
        """
        
        return regex.search(pattern=pattern, string=string, flags=regex.I) 

    def filter_papers(self, grouped_stems):
        """
        Filters papers in the anthology based on specified stems, selecting those published in or after 2018. Each group of keyword stems must be represented in either the title or abstract of the selected papers.

        Args:
            grouped_stems (list[list[str]]): A list where each item is a list of stemmed keywords representing a different thematic group.
        """
        for id_, paper in self.anthology.papers.items():
            
            original_abstract = paper.get_abstract('text')
            original_title = paper.get_title('text')

            abstract_stem = " ".join(self.stemmer.stem_text(original_abstract))
            title_stem = " ".join(self.stemmer.stem_text(original_title))
            
            # Check for the presence of at least one keyword from each group in either the title or the abstract
            if all(
                any(self.search_regex(stem, abstract_stem) or self.search_regex(stem, title_stem) for stem in group.split("|"))
                for group in grouped_stems
            ):
                paper_info = paper.as_dict()
                year = int(paper_info.get('year'))

                if year >= 2018:
                    self.selected_papers[id_] = {
                        'title': original_title,
                        'abstract': original_abstract,
                        'url': paper_info.get('url')
                    }
            

if __name__ == "__main__":
    # Initialize the Stemmer and Anthology
    stemmer = Stemmer()
    anthology = Anthology(importdir='acl-anthology/data')

    # Define search strings for different topics
    topic_keywords = {
        "sentiment": "sentiment|emotion",
        "finance": "financial|market|asset|stock|finance",
        "prediction": "prediction|forecasting|impact|influence|correlation"
    }

    # Stem the search strings
    stemmed_keywords = {topic: stemmer.stem_keywords(keywords) for topic, keywords in topic_keywords.items()}

    # Initialize the PaperSelector
    selector = PaperSelector(stemmer, anthology)

    selector.filter_papers(stemmed_keywords.values())
    # # Filter papers based on stemmed keywords
    # for stems in stemmed_keywords.values():
    #     selector.filter_papers(stems.split("|"))
    
    print(f"Number of selected papers: {len(selector.selected_papers.keys())}")
    
   # Save the selected papers to a JSON file
    with open('selected_papers.json', 'w', encoding='utf-8') as f:
        json.dump(selector.selected_papers, f, ensure_ascii=False, indent=4)
