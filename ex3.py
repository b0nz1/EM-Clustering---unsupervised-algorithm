"""
Zaidman Igal 311758866
Alon Gadot 305231524
"""
import sys
import numpy
import multiprocessing as mp
import math
from collections import Counter
import matplotlib.pyplot as pl

#CONSTANT PARAMETERS
MIN_FREQ = 3
CLUSTERS = 9
LAMBDA = 0.05
EPS = 0.001
EPS_THRESHOLD = 50
K = 10
    
class ArticleData(object):
    def __init__(self,index,article,article_topics,words):
        article_words = [word for word in article.split() if word in words]
        self.wordCounter = Counter(article_words)
        self.len = len(article_words)
        self.potipcs = article_topics
        self.wt = numpy.zeros(CLUSTERS)
        self.wt[index % CLUSTERS] = 1
class EM(object):
    #def __init__(self,data,wordCounter,topics):
    def __init__(self,articles,article_topics,words,topics):
        self.alpha = None
        self.p = None
        self.topics = topics
        self.words = words
        self.articles =[ArticleData(i,articles[i],article_topics[i],words) for i in range(len(articles))]
     
    def execute(self):
        print("START: EM execution")
        likelihood = []
        perplexity = []
        to_run = True
        epochs = 0
        while to_run:
            # after atleast 3 epochs check if log liklihood did not improve
            if (epochs >= 3 and (likelihood[-1] - likelihood[-3]) < EPS_THRESHOLD) or epochs == 75:
            #if epochs == 3:#this is a threshold for debugging only 
                to_run = False
            else:
                print("Epoch: " + str(epochs))
                self.M()
                self.E()
                
                curr_likelihood = self.calc_likelihood()
                print("Likelihood: " + str(curr_likelihood))
                curr_perplexity = self.calc_perplexity()
                
                likelihood.append(curr_likelihood)
                perplexity.append(curr_perplexity)
                epochs += 1
                
        #save graphs of likeligood and perplexity adn confusion matrix
        create_graph(likelihood,"likelihood")
        create_graph(perplexity,"perplexity")
        self.create_confusion()
        
    #create and save confusion matrix for report 
    def create_confusion(self):
        matrix = []
        counter = numpy.zeros(CLUSTERS)

        for i in range(CLUSTERS):
            matrix.append({topic: 0 for topic in self.topics})
            for article in self.articles:
                if numpy.argmax(article.wt) == i:
                    counter[i] += 1
                    for topic in article.potipcs:
                        matrix[i][topic] += 1

        lines = [("{0},{1},{2}".format(str(i), (",".join([str(c) for c in row.values()])), counter[i]))
                 for i, row in enumerate(matrix)]
        lines.insert(0, "," + ",".join(self.topics))

        fh = open("matrix.csv", "w")
        fh.write("\n".join(lines))
        fh.close()
        
    #calculate the perplexity
    def calc_perplexity(self):
        print("START perplexity")
        perplex_sun = 0

        for article in self.articles:
            article_pred = numpy.argmax(article.wt)
            prob_sum = 0

            for word, count in article.wordCounter.items():
                numerator = (self.p[article_pred][word] * article.len + LAMBDA)
                denominator = (article.len + len(self.words) * LAMBDA)
                #sum the probabilities of the right classifications
                prob_sum += numpy.log(numerator / denominator) * count

            perplex_sun += numpy.exp(prob_sum / article.len * -1)

        return perplex_sun / len(self.articles)
    
    #calculate the likelihood with underflow
    def calc_likelihood(self):
        print("START: likelihood")
        likelihoods = []
        for article in self.articles:
            z = self.article_z(article)
            m = max(z)
            likelihoods.append(m + math.log(sum([math.exp(z[i] - m) for i in range(CLUSTERS) if z[i] - m >= -1 * K])))

        return sum(likelihoods)
        
    
    def article_z(self,article):
        z = numpy.zeros(CLUSTERS)
        for clust in range(CLUSTERS):
            num = sum([count * numpy.log(self.p[clust][word]) for (word, count) in article.wordCounter.items()])
            z[clust] = numpy.log(self.alphas[clust]) + num
        return z
    
    #execute E step with underflow
    def E(self):
        print("START: E step")
        for article in self.articles:
            z = self.article_z(article)
            m = max(z)

            denominator = sum([math.exp(z[i] - m) for i in range(CLUSTERS) if z[i] - m >= -K])
            for i in range(CLUSTERS):
                article.wt[i] = 0 if z[i] - m < -K else math.exp(z[i] - m) / denominator
                
                #if z[i] - m < -K:
                #    article.wt[i] = 0
                #else:
                #    article.wt[i] = math.exp(z[i] - m) / denominator
                
    #execute the M step with smoothing        
    def M(self):
        print("START: M step")
        alphas = []
        for i in range(CLUSTERS):
            sum_wti = sum([atricle.wt[i] for atricle in self.articles])
            alpha_i = float(sum_wti) / len(self.articles)
            alpha_i = EPS if alpha_i < EPS else alpha_i
            alphas.append(alpha_i)
        sum_alphas = sum(alphas)
        self.alphas = [alpha / sum_alphas for alpha in alphas]
        
        # update p using multiprocessing, process for each cluster
        p = mp.Manager().dict()
        processes = [mp.Process(target=self.calc_pi, args=(i, p)) for i in range(CLUSTERS)]
    
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        self.p = dict(p)
             
    #calculate the Pi in the M step            
    def calc_pi(self,i, p):
        pi = {}
        for word in self.words:
                numerator = 0
                denominator = 0
    
                for article in self.articles:
                    numerator += article.wt[i] * article.wordCounter[word]
                    denominator += article.wt[i] * article.len
    
                pi[word] = (float(numerator) + LAMBDA) / (denominator + (len(self.words) * LAMBDA))
        p[i] =pi
        
#create and save the graphs for the perplexity and likelihood
def create_graph(lst, title):
    pl.figure()
    pl.plot(range(len(lst)), lst)
    pl.xlabel('epoch')
    pl.ylabel(title)
    pl.savefig(title + '.png')
        
#load all the articles data
def loadArticles(file_name): 
    print("START: load data")
    articles = []
    article_tags = []
    words = []
    
    file_data = []
    file = open(file_name).read().split("\n")
    for line in file:
        line = line.strip()
        if line != "":
            file_data.append(line)

    i = 0
    while i < len(file_data):
        article_tags.append(file_data[i].rstrip(">").split()[2:])
        articles.append(file_data[i+1])
        for w in file_data[i+1].split():
            words.append(w)
        i += 2 #advance index to next article
        words
    
    wordsCounter = Counter(words)
    wordsFiltered = [w for w in wordsCounter if wordsCounter[w] > MIN_FREQ]
    return articles, article_tags, wordsFiltered

#load the topics from file
def loadTopics(file_name):
    file_data = []
    file = open(file_name).read().split("\n")
    for line in file:
        line = line.strip()
        if line != "":
            file_data.append(line)
    return file_data

if __name__ == "__main__":
    develop_file = sys.argv[1]
    topics_file = sys.argv[2]
    #develop_file = "./dataset/develop.txt"
    #topics_file = "./dataset/topics.txt"
    
    articles,article_topics, words = loadArticles(develop_file)
    print("vocabulary size after filter: " + str(len(words)))
    topics = loadTopics(topics_file)
    
    em_obj = EM(articles,article_topics,words,topics)
    likelihood = []
    perplexity = []
    em_obj.execute()