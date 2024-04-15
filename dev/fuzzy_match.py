import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from ftfy import fix_text
import time
import warnings
import unicodedata
import nmslib

warnings.simplefilter(action='ignore', category=Warning)

# Global Variables for Removing/Replacing Words

bank_stopwords=['biz', 'bv', 'co', 'comp', 'company',
                'corp','corporation', 'dba',
                'inc', 'incorp', 'incorporat',
                'incorporate', 'incorporated', 'incorporation',
                'international', 'intl', 'intnl',
                'limited' ,'llc', 'ltd', 'llp',
                'machines', 'pvt', 'pte', 'private', 'unknown', 
                'bank', 'trust', 'finance', 'financial']

bank_replacements = {
    'bk': 'bank',
    'nb': 'national bank',
    'cty': 'county',
    'tc': 'trust company',
    'na': 'national association',
    'svg': 'savings',
    'st': 'street',
    'cmnty': 'community',
    'cmrl': 'commercial',
    'br': 'branch',
    'bkg': 'banking',
    'intl': 'international',
    'sb': 'savings bank',
    'fsb': 'federal savings bank',
    'ssb': 'state savings bank',
    'mrch': 'merchant',
    'agy': 'agency',
    'co': 'company',
    'op': 'operative'
}
def replace_abbreviations(text):
    """
    Replaces abbreviations found in a given text with their full forms based on a predefined dictionary of banking abbreviations.

    Parameters:
    - text (str): The string of text in which abbreviations are to be replaced.

    Variables:
    - abbr (str): An abbreviation found in the `text` that matches a key in the `bank_replacements` dictionary.
    - full (str): The full form corresponding to an abbreviation, used to replace the abbreviation in the `text`.

    Returns:
    - text (str): The modified text with all recognized abbreviations replaced by their full forms.
    """
    # Iterate through each abbreviation and its full form in the bank_replacements dictionary
    for abbr, full in bank_replacements.items():
        # Replace the abbreviation in the text with its full form, ensuring the abbreviation is surrounded by spaces
        # to avoid partial matches within words. Spaces are included in the search and replace strings to ensure
        # that only whole words are replaced.
        text = text.replace(f" {abbr} ", f" {full} ")
    # Return the text after all abbreviations have been replaced
    return text

def filter_ascii(text):
    """
    Filters out non-ASCII characters from the text by converting it to a normalized form where non-ASCII characters 
    are represented as their closest ASCII equivalents, and then encoding and decoding to remove any remaining non-ASCII characters.

    Parameters:
    - text (str): The string of text from which non-ASCII characters need to be removed.

    Returns:
    - text (str): The ASCII-only version of the input text.

    Explanation:
    - The function first normalizes the text using the 'NFKD' form with unicodedata.normalize, which separates characters 
      into their decomposed form. This step helps in converting characters with accents into their ASCII equivalents.
    - Then, it encodes the normalized string to ASCII, ignoring errors, which effectively removes any non-ASCII characters.
    - Finally, it decodes the encoded bytes object back to a string, specifying 'utf-8' encoding and ignoring errors, 
      resulting in a string that contains only ASCII characters.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_special_characters(text, remove_digits=False):
    """
    Removes special characters from a given text, with an option to also remove digits. This function is useful
    for cleaning and standardizing textual data, especially when preparing text for processing or analysis.

    Parameters:
    - text (str): The string from which special characters (and optionally digits) are to be removed.
    - remove_digits (bool, optional): If True, digits will also be removed from the text. Defaults to False.

    Returns:
    - text (str): The cleaned text with special characters (and optionally digits) removed.

    Explanation:
    - The function uses regular expressions to define a pattern for what to keep in the text. By default, it keeps
      letters (both uppercase and lowercase), digits, and whitespace, effectively removing all special characters.
    - If `remove_digits` is set to True, the pattern changes to exclude digits as well, keeping only letters and whitespace.
    - It first removes special characters (and digits if specified) by replacing matches of the pattern with an empty string.
    - It then replaces newline characters (`\\n`) with a space to ensure the cleaned text does not contain any newline characters,
      which could interfere with further processing or analysis.
    """
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = re.sub('\\n',' ',text)
    return text


def clean_stopwords(text, eng=False):
    """
    Removes stopwords from the text. This function is particularly useful for cleaning and preprocessing textual data,
    such as company names, by removing common but uninformative words. It supports removing a custom list of banking-specific
    stopwords, with an option to include general English stopwords as well.

    Parameters:
    - text (str): The string from which stopwords are to be removed.
    - eng (bool, optional): If True, both English stopwords and banking-specific stopwords will be removed from the text.
      Defaults to False, in which case only banking-specific stopwords are removed.

    Returns:
    - text (str): The text with stopwords removed.

    Explanation:
    - The function defines a custom list of stopwords specific to banking and corporate entities (`bank_stopwords`). 
      If `eng` is True, it extends this list with a predefined list of general English stopwords (`ENGLISH_STOP_WORDS`).
    - It then iterates through each stopword in the combined list and removes occurrences of these stopwords from the text.
      This is done using regular expressions that match whole words (`\b` denotes a word boundary) to ensure that only complete
      words are removed and not parts of words.
    - This cleaning step helps in focusing on the more meaningful parts of textual data by removing noise associated with common
      but uninformative words.
    """
    if eng == False:
        custom = bank_stopwords
    else:
        custom = bank_stopwords + list(ENGLISH_STOP_WORDS)
    for x in custom:
        pattern2 = r'\b'+x+r'\b'
        text = re.sub(pattern2, '', text)
    return text

def clean_spaces(text):
    """
    Cleans the input text by removing extra spaces. If the text becomes too short (e.g., empty) after cleaning, 
    it replaces it with a placeholder text. This function is useful for ensuring that text data is neatly formatted 
    and ready for further processing or analysis.

    Parameters:
    - text (str): The string from which extra spaces are to be removed.

    Returns:
    - text (str): The cleaned text with extra spaces removed. If the text is too short after cleaning, returns 'Tooshorttext'.

    Explanation:
    - The function first replaces instances of double spaces with a single space to remove unnecessary extra spaces.
    - It then strips leading and trailing spaces from the text to ensure it is neatly trimmed.
    - If this process results in a string with a length less than 1 (i.e., an empty string), the function returns a placeholder 
      'Tooshorttext' to indicate that the input was too short after cleaning. This helps in maintaining consistency in datasets 
      where empty or nearly empty strings are undesirable.
    """
    text = text.replace('  ', ' ')
    text = text.strip()
    if len(text) < 1:
        text = 'Tooshorttext'
    return text

def preprocess_text(column, remove_digits=False, eng=False):
    """
    Processes a column of textual data by applying a series of cleaning functions to each element. This function is
    designed to standardize and clean text data, such as company names, before analysis or matching.

    Parameters:
    - column (iterable of str): An iterable (e.g., list or pandas Series) containing the text to be preprocessed.
    - remove_digits (bool, optional): If True, digits are removed from the text as part of the cleaning process.
                                      Defaults to False.
    - eng (bool, optional): If True, common English stopwords are removed in addition to banking-specific stopwords.
                            Defaults to False.

    Returns:
    - column (list of str): A list of cleaned and standardized text.

    Explanation:
    - The function sequentially applies several preprocessing steps to each element in the input column:
      1. `filter_ascii` to remove non-ASCII characters.
      2. `remove_special_characters` to remove special characters and optionally digits.
      3. Converts text to lowercase.
      4. `replace_abbreviations` to replace common banking abbreviations with their full forms.
      5. `clean_stopwords` to remove stopwords.
      6. `clean_spaces` to trim extra spaces and ensure the text does not have double spaces or leading/trailing spaces.
    - This series of steps ensures that the text data is clean, standardized, and ready for further processing or analysis.
    """
    column = [filter_ascii(text) for text in column]
    column = [remove_special_characters(text, remove_digits) for text in column]
    column = [text.lower() for text in column]
    column = [replace_abbreviations(text) for text in column]
#     column = [clean_stopwords(text, eng) for text in column]
    column = [clean_spaces(text) for text in column]

    return column

def ngrams(string, n=3):
    """
    Generates a list of n-grams from the input string after applying various preprocessing steps to normalize and clean the text.

    Parameters:
    - string (str): The input text from which n-grams are to be generated.
    - n (int, optional): The length of each n-gram. Default is 3.

    Variables:
    - string (str): Used iteratively, holds the text as it is transformed through various preprocessing steps.
    - chars_to_remove (list of str): Specifies characters to be removed from the text during preprocessing.
    - rx (str): A regex pattern built from `chars_to_remove` for identifying characters to remove from the text.
    - ngrams (zip object): An iterable that generates tuples representing each n-gram by zipping together shifted versions of the string.

    Returns:
    - list of str: A list where each element is a string representing an n-gram of the specified length, derived from the preprocessed input text.
    """

    # Ensure input is string format
    string = str(string)
    # Convert string to lowercase for normalization
    string = string.lower()
    # Fix text for encoding or special character issues
    string = fix_text(string)
    # Remove non-ASCII characters
    string = string.encode("ascii", errors="ignore").decode()
    
    # Define characters to remove from the text
    chars_to_remove = [")","(",".","|","[","]","{","}","'","-"]
    # Create regex pattern for characters to be removed
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    # Remove specified characters from the string
    string = re.sub(rx, '', string)
    
    # Standardize '&' to 'and' for consistency
    string = string.replace('&', 'and')
    # Abbreviate 'limited' to 'ltd'
    string = string.replace('limited', 'ltd')
    # Normalize case to title case for consistency
    string = string.title()
    
    # Remove extra spaces, ensuring single spaces between words
    string = re.sub(' +',' ',string).strip()
    # Pad names with spaces to facilitate n-gram generation
    string = ' '+ string +' '
    
    # Generate n-grams
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def tfidf_fuzzy_match(main_df, main_name_coln, main_id_coln, match_df, match_name_coln, match_id_coln, compare_initials = False):
    """
    Performs fuzzy matching of company names between two dataframes using TF-IDF vectorization and nearest neighbor search.

    Parameters:
    - main_df (pd.DataFrame): DataFrame containing the primary dataset for matching.
    - main_name_coln (str): Column name in `main_df` for the company names.
    - main_id_coln (str): Column name in `main_df` for the company identifiers.
    - match_df (pd.DataFrame): DataFrame containing the dataset to match against.
    - match_name_coln (str): Column name in `match_df` for the company names.
    - match_id_coln (str): Column name in `match_df` for the company identifiers.
    - compare_initials (bool, optional): Flag to determine if an additional check should be performed to match the first initials of each word in the company names. Defaults to False.

    Variables:
    - match_names (list of str): Unique company names from `match_df`.
    - match_ids (list of str): Unique company identifiers from `match_df`.
    - vectorizer (TfidfVectorizer): TF-IDF vectorizer object for converting text to vector form.
    - match_tf_idf_matrix (sparse matrix): TF-IDF matrix for `match_names`.
    - main_names (list of str): Unique company names from `main_df`.
    - main_tf_idf_matrix (sparse matrix): TF-IDF matrix for `main_names`.
    - data_matrix (sparse matrix): Copy of `match_tf_idf_matrix` for indexing.
    - index (nmslib.index): NMSLIB index for efficient similarity search.
    - nbrs (list of tuples): List of tuples containing the index of the nearest neighbor and the corresponding similarity score for each name in `main_df`.
    - mts (list of lists): Matches found, each element is a list containing details of the match.

    Returns:
    - mts (pd.DataFrame): DataFrame containing the matches found, including company names, identifiers, and similarity scores.

    Explanation:
    Each step in the function is commented below for clarity.
    """
    # Extract unique names and IDs from match_df to prepare for matching
    match_names = list(match_df[match_name_coln].unique())
    match_ids = list(match_df[match_id_coln].unique())
    
    # Initialize TF-IDF vectorizer with custom n-gram analyzer
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    # Vectorize match_df names into TF-IDF matrix for similarity comparison
    match_tf_idf_matrix = vectorizer.fit_transform(match_names)

    # Extract unique names from main_df for matching
    main_names = list(main_df[main_name_coln].unique())
    # Transform main_df names into TF-IDF matrix using the same vectorizer
    main_tf_idf_matrix = vectorizer.transform(main_names)

    ## Unique names from both main_df and match_df are extracted and vectorized using the TF-IDF 
    ## which transforms text into a meaningful representation of numbers which is used to fit machine algorithm models.
    data_matrix = match_tf_idf_matrix#[0:1000000]
    # Set index parameters
    # These are the most important ones
    M = 80
    efC = 1000

    num_threads = 4 # adjust for the number of threads
    # Initialize the nearest neighbor search index with specified method and space
    index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 
    index.addDataPointBatch(data_matrix) # Add match_df TF-IDF matrix to the index

    # Time the index creation for performance evaluation
    start = time.time()
    index.createIndex() # Create the index for nearest neighbor search
    end = time.time() 
    print('Indexing time = %f' % (end-start))
    
    # Perform k-nearest neighbor search with specified number of threads
    num_threads = 4
    K = 1  # Number of nearest neighbors to find (1 for the closest match)
    query_matrix = main_tf_idf_matrix # TF-IDF matrix of main_df names to query against the index
    start = time.time() 
    query_qty = query_matrix.shape[0]  # Number of queries to process
    nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads) # Execute the queries
    end = time.time() 
    print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
          (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty))

    # Initialize list to hold match results
    mts =[]
    for i in range(len(nbrs)):
        main_name = main_names[i] # Name from main_df being matched
        # Find the ID in main_df corresponding to the main_name
        main_id = main_df.loc[main_df.index[main_df[main_name_coln] == main_name].tolist()[0], main_id_coln]
        try:
            candidate_name = match_names[nbrs[i][0][0]] # Matched name from match_df
            conf = nbrs[i][1][0] # Similarity score/confidence of the match
            # Compare initials only if specified to do so
            if compare_initials == True:
                ## After finding the nearest neighbor based on TF-IDF similarity, the script performs an additional check to ensure that the first letters of each corresponding word in the matched names are the same. 
                ## This step is intended to filter out matches that are textually similar but may not actually refer to the same entity due to differences in initials
                # Split names into words and get the first letter of each word
                main_name_initials = [word[0].lower() for word in main_name.split()]
                candidate_name_initials = [word[0].lower() for word in candidate_name.split()]

                # Determine the length for comparison based on the shorter name
                compare_length = min(len(main_name_initials), len(candidate_name_initials))

                # Check if the first letters of each corresponding word are the same
                if main_name_initials[:compare_length] == candidate_name_initials[:compare_length]:
                    matched_name = candidate_name
                    matched_id = match_df.loc[match_df.index[match_df[match_name_coln] == matched_name].tolist()[0], match_id_coln]
                else:
                    raise ValueError("First letters of words do not match")
            elif compare_initials == False:
                matched_name = candidate_name
                matched_id = match_df.loc[match_df.index[match_df[match_name_coln] == matched_name].tolist()[0], match_id_coln]

        except:
            matched_name = "no match found"
            matched_id = None
            conf = None
        # Append the match details to the results list
        mts.append([main_name, main_id, matched_name, matched_id, conf])

    # Convert match results list to DataFrame for easy handling and return it
    mts = pd.DataFrame(mts,columns=[main_name_coln, main_id_coln, match_name_coln, match_id_coln, 'conf'])

    return mts


def gen_match_links(main_df, main_name_coln, main_id_coln, match_df, match_name_coln, match_id_coln, compare_initials = False, conf_threshold = 0.7, year_range = [2010, 2023]):
    """
    Matches records between two dataframes based on fuzzy matching of company names for each quarter of each year within a specified range.

    Parameters:
    - main_df (pd.DataFrame): The main DataFrame containing company names and other details to be matched.
    - main_name_coln (str): The column name in `main_df` that contains company names.
    - main_id_coln (str): The column name in `main_df` that contains unique identifiers for each company.
    - match_df (pd.DataFrame): The DataFrame with which the records from `main_df` are to be matched.
    - match_name_coln (str): The column name in `match_df` that contains company names.
    - match_id_coln (str): The column name in `match_df` that contains unique identifiers for each company.
    - conf_threshold (float, optional): The confidence threshold for considering a match to be good enough. Defaults to 0.7.
    - year_range (list, optional): A list specifying the start and end years for the range within which to perform the matching. Defaults to [2010, 2023].

    Variables:
    - final_XW (pd.DataFrame): An empty DataFrame initialized to store the final matched records.
    - year (int): The current year being processed in the loop.
    - quarter (int): The current quarter being processed in the loop.
    - main_df_one_q (pd.DataFrame): A filtered DataFrame from `main_df` containing only records for a specific year and quarter.
    - match_df_one_q (pd.DataFrame): A filtered DataFrame from `match_df` containing only records for the same specific year and quarter as `main_df_one_q`.
    - [column_name]_cleaned (pd.Series): A Series added to either `main_df_one_q` or `match_df_one_q` containing preprocessed (cleaned and standardized) company names for matching.
    - match_link_one_q (pd.DataFrame): The result of fuzzy matching between cleaned company names in `main_df_one_q` and `match_df_one_q`.
    - good_match_one_q (pd.DataFrame): A subset of `match_link_one_q` containing only matches that meet or exceed the confidence threshold.

    Returns:
    - final_XW (pd.DataFrame): A DataFrame containing all records that were matched with a confidence level above the threshold, across all specified years and quarters.
    """
    # Initializes an empty DataFrame to store final matched records
    final_XW = pd.DataFrame()

    # Iterates through each year within the specified range
    for year in range(year_range[0], year_range[1] + 1):
        # Iterates through each quarter of the year
#         for quarter in range(1, 5):
        # Filters `main_df` for the current year and quarter
#             main_df_one_q = main_df.loc[(main_df['year'] == year) & (main_df['quarter'] == quarter)]
        main_df_one_q = main_df.loc[(main_df['year'] == year)]

        # Filters `match_df` for the current year and quarter
#             match_df_one_q = match_df.loc[(match_df['year'] == year) & (match_df['quarter'] == quarter)]
        match_df_one_q = match_df.loc[(match_df['year'] == year)]

        # Pre-processes company names in both dataframes to clean and standardize them for better matching
        main_df_one_q[main_name_coln + '_cleaned'] = preprocess_text(main_df_one_q[main_name_coln])
        match_df_one_q[match_name_coln + '_cleaned'] = preprocess_text(match_df_one_q[match_name_coln])

        try:
            # Performs fuzzy matching using TF-IDF between cleaned company names from both dataframes
            match_link_one_q = tfidf_fuzzy_match(main_df_one_q, main_name_coln + '_cleaned', main_id_coln, match_df_one_q, match_name_coln + '_cleaned', match_id_coln, compare_initials = compare_initials)

            # Filters matches that meet the confidence threshold (i.e., the match is considered good enough)
            good_match_one_q = match_link_one_q.loc[match_link_one_q['conf'] < -conf_threshold]
            # Adds year and quarter information to the good matches for tracking
            good_match_one_q['year'] = year
#                 good_match_one_q['quarter'] = quarter

            # Logs the number of matches found for this quarter
            print("{} banks matched for {}".format(len(good_match_one_q), year))

            # Appends the good matches for this quarter to the final DataFrame
            final_XW = pd.concat([final_XW, good_match_one_q], axis = 0, ignore_index=True)
        except:
            # Logs an error message if there's an issue during the matching process (e.g., empty dataframes)
            print("No datasets for {}".format(year))
            pass
    # Returns the final DataFrame containing all good matches across all specified years and quarters
    return final_XW