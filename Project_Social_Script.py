
import pandas as pd
import os
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from operator import itemgetter
import re
import itertools
import collections
import nltk
from nltk import bigrams
from nltk.corpus import stopwords

# Diffusion Network
def createDataFrame(type):
    '''This function helps in creating a panda dataframe as it is easy to use that for social network'''
    tweets_dataframe = pd.read_json(type + 'Tweets.json')
    tweets_new_df = pd.DataFrame(columns = ["created_at", "id", "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_user_id",
                                      "retweeted_id", "retweeted_screen_name", "user_mentions_screen_name", "user_mentions_id", 
                                     "user_id", "screen_name", "followers_count"])
    # Columns that are going to be the same
    equal_columns = ["created_at", "id", "full_text"]
    tweets_new_df[equal_columns] = tweets_dataframe[equal_columns]
    return tweets_dataframe,tweets_new_df

 
def get_basics(tweets_new_df,tweets_dataframe):
    '''Get the basic information about user'''
    tweets_new_df["screen_name"] = tweets_dataframe["user"].apply(lambda x: x["screen_name"])
    tweets_new_df["user_id"] = tweets_dataframe["user"].apply(lambda x: x["id"])
    tweets_new_df["followers_count"] = tweets_dataframe["user"].apply(lambda x: x["followers_count"])
    return tweets_new_df

def get_usermentions(tweets_new_df,tweets_dataframe):
    '''Get the user mentions'''
    tweets_new_df["user_mentions_screen_name"] = tweets_dataframe["entities"].apply(lambda x: x["user_mentions"][0]["screen_name"] if x["user_mentions"] else np.nan)
    tweets_new_df["user_mentions_id"] = tweets_dataframe["entities"].apply(lambda x: x["user_mentions"][0]["id_str"] if x["user_mentions"] else np.nan)
    return tweets_new_df

def get_retweets(tweets_new_df,tweets_dataframe):
    '''Get Retweets Data'''  
    tweets_new_df["retweeted_screen_name"] = tweets_dataframe["retweeted_status"].apply(lambda x: x["user"]["screen_name"] if x is not np.nan else np.nan)
    tweets_new_df["retweeted_id"] = tweets_dataframe["retweeted_status"].apply(lambda x: x["user"]["id_str"] if x is not np.nan else np.nan)
    return tweets_new_df

def get_in_reply(tweets_new_df,tweets_dataframe):
    '''Get the information about replies'''
    tweets_new_df["in_reply_to_screen_name"] = tweets_dataframe["in_reply_to_screen_name"]
    tweets_new_df["in_reply_to_status_id"] = tweets_dataframe["in_reply_to_status_id"]
    tweets_new_df["in_reply_to_user_id"]= tweets_dataframe["in_reply_to_user_id"]
    return tweets_new_df

def fill_df(tweets_new_df,tweets_dataframe):
    '''Create a Useful dataframe that will help in diffusion network'''
    get_basics(tweets_new_df,tweets_dataframe) #This helps in getting basic info of users
    get_usermentions(tweets_new_df,tweets_dataframe) #This will help in getting the usermentions
    get_retweets(tweets_new_df,tweets_dataframe) #This will help in getting the retweet data
    get_in_reply(tweets_new_df,tweets_dataframe) #This will help in gettin the reply data of the tweet
    return tweets_new_df


def get_interactions(row):
    '''Get the interactions between the different users'''
    user = row["user_id"], row["screen_name"]
    if user[0] is None:
        return (None, None), []
    
    interactions = set()
    
    # Interactions corresponding to replies
    interactions.add((row["in_reply_to_user_id"], row["in_reply_to_screen_name"]))
    # Interactions with retweets
    interactions.add((row["retweeted_id"], row["retweeted_screen_name"]))
    # Interactions with user mentions
    interactions.add((row["user_mentions_id"], row["user_mentions_screen_name"]))
    
    #Data Cleaning
    interactions.discard((row["user_id"], row["screen_name"]))
    interactions.discard((None, None))
    return user, interactions

def fill_dataframe(type):
    '''Getting a useful interation dataframe from the tweets fetched'''
    tweets_dataframe,tweets_new_df=createDataFrame(type)
    tweets_new_df = fill_df(tweets_new_df,tweets_dataframe)
    tweets_new_df = tweets_new_df.where((pd.notnull(tweets_new_df)), None)
    return tweets_new_df

def create_graph(tweets_new_df):
    '''Creating the DiffusionNetworkGraph'''
    graph = nx.Graph()
    for index, tweet in tweets_new_df.iterrows():
        user, interactions = get_interactions(tweet)
        user_id, user_name = user
        tweet_id = tweet["id"]
        for interaction in interactions:
            int_id, int_name = interaction
            graph.add_edge(user_id, int_id, tweet_id=tweet_id)
            graph.nodes[user_id]["name"] = user_name
            graph.nodes[int_id]["name"] = int_name 

    print(f"There are {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges present in the Graph")
    degrees = [val for (node, val) in graph.degree()]
    print(f"The maximum degree of the Graph is {np.max(degrees)}")   
    print(f"The minimum degree of the Graph is {np.min(degrees)}")
    print(f"The average degree of the nodes in the Graph is {np.mean(degrees):.1f}")    
    print(f"The most frequent degree of the nodes found in the Graph is {stats.mode(degrees)[0][0]}")
    if nx.is_connected(graph):
        print("The graph is connected")
    else:
        print("The graph is not connected")
    print(f"There are {nx.number_connected_components(graph)} connected components in the Graph")
    return graph

def create_largest_connected_graph(graph):
    '''Create the largest connected graph'''
    A=list(graph.subgraph(c) for c in nx.connected_components(graph))
    largest_subgraph = max(A, key=len)
    print(f"There are {largest_subgraph.number_of_nodes()} nodes and {largest_subgraph.number_of_edges()} edges present in the largest component of the Graph")
    if nx.is_connected(largest_subgraph):
        print("The graph is connected")
    else:
        print("The graph is not connected")
    print(f"The diameter of our Graph is {nx.diameter(largest_subgraph)}")
    print(f"The average distance between any two nodes is {nx.average_shortest_path_length(largest_subgraph):.2f}")
    return largest_subgraph

def calculating_network_measures(graph):
    '''Calculating different network measures'''
    graph_centrality = nx.degree_centrality(graph)
    max_de = max(graph_centrality.items(), key=itemgetter(1))
    graph_closeness = nx.closeness_centrality(graph)
    max_clo = max(graph_closeness.items(), key=itemgetter(1))
    graph_betweenness = nx.betweenness_centrality(graph, normalized=True, endpoints=False)
    max_bet = max(graph_betweenness.items(), key=itemgetter(1))
    page_rank = nx.pagerank(graph)
    max_page_rank = max(page_rank.items(), key=itemgetter(1))
    print(f"the node with id {max_de[0]} has a degree centrality of {max_de[1]:.2f} which is the maximum of the Graph")
    print(f"the node with id {max_clo[0]} has a closeness centrality of {max_clo[1]:.2f} which is the maximum of the Graph")
    print(f"the node with id {max_bet[0]} has a betweenness centrality of {max_bet[1]:.2f} which is the maximum of the Graph")
    print(f"the node with id {max_page_rank[0]} has a page rank of {max_page_rank[1]:.2f} which is the maximum of the Graph")
    return [max_de[0],max_clo[0],max_bet[0]]

def plot_graph(type,graph):
    '''To plot the diffusion network graph'''
    plt.figure(figsize = (20,20))
    nx.draw(graph, edge_color="black", linewidths=0.3,node_size=60, alpha=0.6, with_labels=False)
    plt.savefig(type+'/'+type+'DiffusionNetworkGraph.png')
    

def plot_large_graph(type,central_nodes,largest_subgraph):
    '''To plot the diffusion network largest subgraph'''
    node_and_degree = largest_subgraph.degree()
    colors_central_nodes = ['red','green','blue']
    pos = nx.spring_layout(largest_subgraph, k=0.05)
    plt.figure(figsize = (20,20))
    nx.draw(largest_subgraph, pos=pos, node_color=range(largest_subgraph.number_of_nodes()), cmap=plt.cm.PiYG, edge_color="black", linewidths=0.3, node_size=60, alpha=0.6, with_labels=False)
    nx.draw_networkx_nodes(largest_subgraph, pos=pos, nodelist=central_nodes, node_size=300, node_color=colors_central_nodes)
    plt.savefig(type+'/'+type+'DiffusionNetworkLargestConnectedGraph.png')
    

def plot_degree_dist(type,G):
    '''To Plot the degree distribution of the graph'''
    degrees = [G.degree(n) for n in G.nodes()]
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.hist(degrees)
    plt.savefig(type+'/'+type+'DegreeDistribution.png')


#Word Co-Occurence Graph 
def remove_url(txt):
    '''Replace URLs found in a text string with nothing'''
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    no_url = url_pattern.sub(r'', txt)
    return no_url

def create_word_occurence_graph(type,tweets_new_df):
    '''Create word Occurence graph'''
    tweets_no_urls = [remove_url(tweet) for tweet in list(tweets_new_df.full_text)]
    words_in_tweet = [tweet.lower().split() for tweet in tweets_no_urls]
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]
    terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw]
    # Flatten list of bigrams in clean tweets
    bigrams_list = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrams_list)

    bigram_counts.most_common(20)
    bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                             columns=['bigram', 'count'])
    # Create dictionary of bigrams and their counts
    d = bigram_df.set_index('bigram').T.to_dict('records')
    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    central_nodes = calculating_network_measures(G)
    fig, ax = plt.subplots(figsize=(20, 10))

    pos = nx.spring_layout(G, k=1.5)
    # Plot networks
    nx.draw_networkx(G, pos,
                    font_size=12,
                    width=3,
                    edge_color='grey',
                    node_color='purple',
                    with_labels = False,
                    ax=ax)
    nx.draw_networkx_nodes(G,pos=pos, nodelist=central_nodes, node_size=300, node_color=['red','green','blue'])

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.045
        ax.text(x, y,
                s=key,
                bbox=dict(alpha=0.25),
                horizontalalignment='center', fontsize=13)
    plt.savefig(type+'/'+type+'WordOccurenceGraph.png')  

    return G
    



def social_network_graphs(type):
    if not os.path.exists(type):
        os.makedirs(type)
    tweets_new_df = fill_dataframe(type)

    # Creating the Graph
    print('------------------------------------------------')
    print('Diffusion Network Graph for ',type)
    print('------------------------------------------------')
    graph = create_graph(tweets_new_df)
    plot_degree_dist(type,graph)
    calculating_network_measures(graph)
    plot_graph(type,graph)

    #Creating the Largest Connected SubGraph
    print('------------------------------------------------')
    print('Largest SubGraph for ',type)
    print('------------------------------------------------')
    largest_subgraph = create_largest_connected_graph(graph)
    largest_central_nodes = calculating_network_measures(largest_subgraph)
    plot_large_graph(type,largest_central_nodes,largest_subgraph)

    #Create Word Occurence Graph
    print('------------------------------------------------')
    print('Word Occurence Network Graph for ',type)
    print('------------------------------------------------')
    create_word_occurence_graph(type,tweets_new_df)

if __name__ == "__main__":
    print('Choose which data to fetch for - AntiVaccine(A) or ProVaccine(B)')
    choice = input()
    print(choice)
    if choice == "A" or choice == "a":
        social_network_graphs('AntiVaccine')
    elif choice == "B" or choice == "b":
        social_network_graphs('ProVaccine')
    else:
        print('Incorrect Choice. Enter A or B')
    



