import math
from math import sqrt

# remove duplicates
def remove_duplicates(s):
  l = s.split()
  k = []

  # for each term in l
  for i in l:
    # check if term exists or occurs more than once and if it is not already in k append it to it
    if (s.count(i)>=1 and (i not in k)):
      k.append(i)

  return ' '.join(k)


# tf calculating method
def calc_tf(t, d):
  # gets the number of times term t appears in document d
  times_t_appears = d.lower().split().count(t.lower())
  # then gets total number of documents
  total_ts_in_doc = len(d.split())
  # then returnes calculated tf
  return times_t_appears / total_ts_in_doc


# Idf calculating method
def calc_idf(t, docs):
  docs_containing_term = 0

  # checks how much documents contain this term t
  for doc in docs:
    if t.lower() in doc.lower():
      docs_containing_term += 1
  
  # and with it then depending if term is contained atleast once the idf is calculated and returned, the + 1.0 required to not have 0.0 returned
  if docs_containing_term > 0:
    return 1.0 + math.log2(len(docs) / docs_containing_term)
  else:
    return 1.0


# tf - Idf calculating method
def calc_tfidf(t, d, docs):
  return calc_tf(t, d) * calc_idf(t, docs)


# method for calculating similarity between doc and query using cosine similarity
def calc_similarity(q, d, docs):

  all_terms = q + " " + d # concatinate query string and document string
  unique_terms = remove_duplicates(all_terms).split() # then remove all duplicates

  # create the document and query vector and assign them the same length as the number of unique terms
  # later in the for loop each will be filled with calculated tf - idf values for each term while taking into account if it is done for the document vector or query vector
  document_vector = [0] * len(unique_terms)
  query_vector = [0] * len(unique_terms)

  # initializing variables that will later be used to calculate the similarity
  dot_product = 0
  sum_of_document_vector_squared_components = 0
  sum_of_query_vector_squared_components = 0

  # this loop will calculate the respective tf - idf for each term, and store these values in both the document vector and query vector
  # of course the values will be different depending if the term is calculated for the document vector or the query vector
  for i in range(len(unique_terms)):
    current_term = unique_terms[i]
    document_vector[i] = calc_tfidf(current_term, d, docs)
    query_vector[i] = calc_tfidf(current_term, q, docs)

  # this loop calculates the sum of document vector squared components and the sum of query vector squared components required to then later calculate the magnitude of each
  for i in range(len(unique_terms)):
    dot_product += query_vector[i] * document_vector[i]

    # sort of guard, if I was to get rid of if statement, similarities with each doc would greatly decrease, unless query was exactly the same as the document
    # hence, this if statement makes sure that the document vector magnitude is calculated only with the tf - idf of terms in document that are in query
    if query_vector[i] != 0.0:
      sum_of_document_vector_squared_components += pow(document_vector[i], 2)

    sum_of_query_vector_squared_components += pow(query_vector[i], 2)

  # calculate and return similarity by dividing the dot product by the document vector magnitude that is multiplied by the query vector magnitude
  if sum_of_document_vector_squared_components != 0.0:
    return (dot_product / (sqrt(sum_of_query_vector_squared_components) * sqrt(sum_of_document_vector_squared_components)))
  else:
    return dot_product / sqrt(sum_of_query_vector_squared_components)


# test method, that uses the calc_similarity function to print out the similarity of specified document and query
def calc_similarity_test():

  # create desired documents (how much you want)
  d1 = "shipment of gold damaged in a fire"
  d2 = "delivery of silver arrived in a silver truck"
  d3 = "shipment of gold arrived in a truck"

  # add all the documents to a collection
  collection = [d1, d2, d3]

  # create a query
  query = "gold damaged"

  # run the calc_similarity function to get the similarity (in this case between query and document 1)
  print(calc_similarity(query, d1, collection))


calc_similarity_test()