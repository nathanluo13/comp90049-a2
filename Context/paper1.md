Automatic Quality Assessment of Wikipedia Articles—A Systematic Literature Review

Authors
PEDRO MIGUEL MOÁS, CARLA TEIXEIRA LOPES
Affiliations
Faculdade de Engenharia da Universidade do Porto, INESC TEC
Published in
ACM Comput. Surv.
Format
Paper
Published
2023-11-01
DOI
10.1145/3625286

Meta Summary
This document is a systematic literature review focusing on the automated evaluation of Wikipedia article quality. It synthesizes research on machine learning algorithms, feature sets, and datasets used to address the challenges of maintaining content standards in a massive, collaboratively edited encyclopedia.

Summary
Wikipedia is the world’s largest online encyclopedia, but maintaining article quality through collaboration is challenging. Wikipedia designed a quality scale, but with such a manual assessment process, many articles remain unassessed. We review existing methods for automatically measuring the quality of Wikipedia articles, identifying and comparing machine learning algorithms, article features, quality metrics, and used datasets, examining 149 distinct studies, and exploring commonalities and gaps in them. The literature is extensive, and the approaches follow past technological trends. However, machine learning is still not widely used by Wikipedia, and we hope that our analysis helps future researchers change that reality.

# Abstract

Wikipedia is the world’s largest online encyclopedia, but maintaining article quality through collaboration is challenging. Wikipedia designed a quality scale, but with such a manual assessment process, many articles remain unassessed. We review existing methods for automatically measuring the quality of Wikipedia articles, identifying and comparing machine learning algorithms, article features, quality metrics, and used datasets, examining 149 distinct studies, and exploring commonalities and gaps in them. The literature is extensive, and the approaches follow past technological trends. However, machine learning is still not widely used by Wikipedia, and we hope that our analysis helps future researchers change that reality.

# 1 INTRODUCTION

Wikipedia is the largest and most well-known online encyclopedia and has kept its growing pace for years. As of April 2023, it contains over 6.6 million English articles, with versions across a list of 321 active languages.

Not only is Wikipedia free, but it is also fully managed by human volunteers, averaging contributions at a rate of 5.7 edits per second during 2022. Its reader base is growing steadily, considering that, last year, Wikipedia totaled close to 280 billion page views across 2 billion unique devices. Some studies even show that most search engines frequently include Wikipedia pages in their results. According to Vincent and Hecht, 80% of common (frequent) queries and 70% of trending queries (news/events) return results from Wikipedia on the first page, when tested with search engines like Google, Bing, and DuckDuckGo.

The fully collaborative aspect of Wikipedia brings its own set of challenges too, as the lack of centralized authority over the editors makes it challenging to ensure quality throughout the website. Only 8.1% of edits are reverted, showing that vandalism and the so-called revert wars are relatively uncommon, but it is still crucial to ensure the improvement of low-quality articles.

Another issue is the substantial quality discrepancy between English Wikipedia and its other versions. First, each non-English version covers a much smaller amount of articles. Also, they are often much more incomplete as well. Roy et al. demonstrate that English articles from Wikipedia are usually longer than their translations. They determined that German articles are, on average, 30% shorter than English ones, and for Spanish articles, that value increases to 47%, but there are still some English articles that are much less complete than their non-English counterparts. Couto and Lopes have also shown this quality discrepancy, although only focused on health-related articles. They used a set of metrics to determine that English articles show the best values for quality, ranking much higher than other idioms.

To better monitor and help maintain the quality of the website, Wikipedia designed a quality scale that aims to rate articles within one of nine possible grades, which go from the most incomplete documents (Starts and Stubs) to the most comprehensive, well-written articles (Featured Articles). However, the majority of Wikipedia is made up of lower-quality articles, with Starts and Stubs accounting for more than 80% of its English content. In comparison, the share of Featured Articles and Good Articles is around 0.7%. Nonetheless, these values are not meant to be taken as official ratings but instead for internal use by the contributors. Besides, not every English article is rated, and the non-English versions of Wikipedia that also assess their content will have different quality scales, evaluated with other criteria. For those reasons, Wikipedia users lack a consistent and transparent method for determining the quality of articles.

Our goal is to review proposed methods for automatically measuring the quality of Wikipedia articles. There already exist a couple of studies sharing a similar objective, but they mostly focus on used article features and do not provide a detailed and exhaustive overview of the existing publications. We believe it would be beneficial to dive deeper into the subject of automatic assessment of Wikipedia quality, so we conducted a systematic literature review to assess the state of the art within that topic, examining and comparing existing approaches used to automatically measure article quality. Specifically, we analyzed machine learning methods, article features, quality metrics, datasets, and other common aspects of these approaches, such as multilingual assessment and data visualization/explanation tools for supporting the reader and editor community. With this review, we aim to provide a starting point for future work that aims to understand Wikipedia quality and design automatic methods for measuring it.

We divided this article into eight sections. After this introduction, Sections 1 and 2 provide some insight about Information Quality (IQ). Section 3 details our methodology for the systematic review, and we list its results in the following sections: Section 4 overviews the included papers and the methods they use, Section 5 summarizes used machine learning approaches, and Section 6 analyses applied article features and quality metrics. We summarize and discuss our findings in Section 7, answering the defined research questions. Finally, we conclude this paper in Section 8, where we reflect on our study and examine future work possibilities.

# 2 INFORMATION QUALITY

It is important to design our definition of quality. Hence, we must first answer: What is quality? Can it be objectively quantified? IQ is an extraordinarily researched topic, and there exist numerous attempts to provide a way to calculate it. Lee et al. break down the measurement of IQ into 15 properties, including Accessibility, Believability, Interpretability, Objectivity, Reputation, and Timeliness. Some of these aspects are much easier to assess than others. However, with recent developments in Natural Language Processing (NLP), some works already attempt to evaluate more complex topics such as bias, neutrality, trustworthiness. Although these studies are an inspiration for authors attempting to tackle the topic of this review, we do not intend to include them unless they specifically propose methods for automatically predicting the quality of articles.

Wikipedia has its definition of quality, too. For instance, the English Wikipedia content assessment guidelines indicate that the most outstanding articles must be well-written, comprehensive, well-researched, and follow their style guidelines, which relate to the IQ properties defined by Lee et al. However, that definition may vary even within Wikipedia: according to Jemielniak and Wilamowski, not all language cultures share the same understanding of quality, which is a vital aspect to consider when designing a multilingual solution for quality assessment.

Overall, quality is a subjective property, so it is difficult to design an objective definition for it. However, there are certainly measurable characteristics that people often relate to outstanding quality, and we plan to determine them and their correlation with excellence in written documents, better understanding which are the most effective methods for predicting it.

# 3 METHODOLOGY

This systematic review aims to answer the following research questions:

RQ1. What are the most commonly used methods for the automatic quality assessment of Wikipedia articles?
RQ2. How can machine learning be best applied to predict article quality, and how do different approaches compare?
RQ3. What are the most common article features and quality metrics used to evaluate article quality in Wikipedia? How do these features compare, and how do they affect the performance of automatic assessment methods?
RQ4. Which common themes and gaps are there in the literature concerning this topic, and how can existing studies be improved to increase the adoption of automatic methods for the quality assessment of Wikipedia?

We guided our selection process by the PRISMA statement, which defines a set of guidelines for conducting systematic literature reviews. Our selection included two main selection stages: Database Querying and Citation Tracking. Figure 1 outlines the selection process by detailing the number of publications included in each phase. Initially, we selected a set of records using research databases, and next, we conducted citation tracking on a subset of the initial selection.

## 3.1 Selection through Database Querying

Our initial selection stage comprises 4 phases: Identification, Screening, Eligibility, and Inclusion.

### 3.1.1 Identification
We considered three primary data sources: Google Scholar, ACM Digital Library, and Web of Science. In all of them, we retrieved all results containing “Wikipedia” and “quality” in its title. Searching in the title significantly reduces the number of results (for instance, reduces Google Scholar results by 99.97%), allowing the screening of all the retrieved results. We present the exact query and number of results for each database in Table 1.

All database queries were run on January 24th, 2023, and were restricted to a date range of 2001–2023. We picked a lower limit of 2001 because Wikipedia was launched in that year, so we would unlikely find related articles from an earlier date.

Our Identification phase returned 379 results, as shown in Table 1. We discarded 175 records as duplicates.

![Fig. 1. An overview of the selection process of our review.](https://kindhearted-porcupine-678.convex.cloud/api/storage/d7ccd2cd-55a6-4ed4-b537-1a0b277138aa)


![Table 1. Search Queries Submitted in Each Database](https://kindhearted-porcupine-678.convex.cloud/api/storage/d0fa4b8e-f698-49c4-82f6-8363727b6d72)


### 3.1.2 Screening

Next, publication titles were assessed for possible relevance within the research area. All results that appeared at least marginally related to the quality assessment of Wikipedia proceeded to the next phase. Nearly all results moved forward, as our query was already somewhat strict. From the 204 non-duplicate titles, 193 were considered possibly relevant for our research. For instance, Salutari et al.’s study [125] was one of the results our query retrieved, but we deemed its title (“A Large-Scale Study of Wikipedia Users’ Quality of Experience”) unrelated to our research topic.

In this phase, we also excluded results that did not respect the usual research article format. This includes theses, dissertations, and technical reports, among others. We opted to include pre-prints to avoid the exclusion of potentially insightful papers.


Finally, we scanned the papers’ abstracts, focusing on the research questions. Only studies that propose automatic methods for measuring Wikipedia quality were included. This phase excluded publications experimenting with manual quality assessment approaches within specific sub-fields (e.g., Health) and studies whose abstracts we could not find. We advanced 130 studies to the next phase.


![Table 2. Inclusion Criteria](https://kindhearted-porcupine-678.convex.cloud/api/storage/37e04f7c-d221-4bfb-bb3f-9b8d05c71aec)


![Table 3. Exclusion Criteria](https://kindhearted-porcupine-678.convex.cloud/api/storage/0148ed7a-45d8-4db6-b5d1-a44f52c27fd8)


### 3.1.3 Eligibility


### 3.1.4 Inclusion


## 3.2 Selection through Citation Tracking


To minimize the probability of excluding relevant articles, we run citation tracking [51], searching through the references (backward tracking) and citations (forward tracking) of all included articles to identify potentially useful results.


Naturally, this procedure directly scales with the number of included articles and the respective number of references and citations. We determined that tracking the entire result set would be impractical, so we decided to only perform backward and forward tracking on the most relevant papers. We assessed relevance using a systematized scoring process, where we assigned an integer value from 0 to 10 based on four questions, as listed in Table 4. We performed citation tracking on all results yielding a global relevance score of 4 or higher.


For each article, we manually checked the titles and abstracts of each reference and citation (we obtained citation data from Google Scholar in March of 2023), applying the same criteria used during the first Screening phase.


All relevant results transitioned to the Eligibility phase directly, therefore, will be assessed for inclusion and may end up being re-tracked, given a high enough relevance score. Overall, we performed this process on 92 different publications, which led to the inclusion of 42 new publications. Our systematic literature review included a total of 149 studies.


## 3.3 Data Collection


Throughout every phase of the selection process, we systematically logged all the data we collected and produced.


![Table 4. Citation Tracking: Relevance Scoring Questions](https://kindhearted-porcupine-678.convex.cloud/api/storage/66ea8eb6-daff-4773-9ffd-37072000404c)


Initially, we store the title of every record we gathered during the first Identification phase and assign them a numeric identifier. We also stored abstracts of publications that advanced to that sub-step of the Screening phase. Due to the substantial amount of inspected references and citations (6438), we did not keep any metadata for publications excluded during the Screening phase of Citation Tracking.


We extracted most of the information during the Inclusion phase. We began by collecting relevant metadata of the 149 studies, such as the title, abstracts, keywords, authors, and year of publication. We then gathered study data, namely machine learning algorithms and respective performance, used article features and quality metrics, and dataset information.


All the information we collected is available in a research data repository [106], allowing readers to consult all the raw information we aggregated to display the results. We also provide a spreadsheet version of the dataset, similar to how we present it in this article, simplifying access for those who prefer not to handle the raw data directly.


# 4 OVERVIEW OF INCLUDED ARTICLES


This section provides an overview of the 149 included papers [3–6, 8, 9, 11–37, 39–44, 46–50, 52–64, 66–68, 70–72, 75, 77, 79, 81–95, 97–101, 103–105, 107, 111, 113–116, 119–124, 126, 128–148, 151–155, 157–161, 169, 171–177, 179–182, 184, 186, 187, 189], analyzing used methods and assessing metadata attributes, like publication venues, citation count, authors, and keywords.


## 4.1 Methods


Most papers (102 out of 149) follow one of these quality assessment strategies: **classical learning (CL)** models trained with article features, **deep learning (DL)** methods using full text or features, and metric-based approaches (MB). Many publications also study the correlation of specific features with quality: although they are not concrete automatic methods for quality prediction, we still consider them relevant for the purpose of this study. We summarize this information in Table 5.


### 4.1.1 Actionable Models and Visualization Tools. Designing an effective quality model for Wikipedia greatly assists Wikipedia users, by allowing easier identification of the best and worst articles. However, this does little for editors who wish to improve them. In the context of Explainable AI [178], it is important to create solutions that also suggest improvement paths, like the actionable model proposed by Warncke-Wang [161]. Some studies propose visualization tools that help solve this aspect. For instance, WikiRank [176] provides quality information and popularity stats of articles across many languages. Other studies [15, 26, 36] share a similar goal, although their solutions are much less thorough.

![Table 5. Most Popular Approaches](https://kindhearted-porcupine-678.convex.cloud/api/storage/ce211d1d-9108-446c-ab9a-be81487cdfcf)

![Fig. 2. Most commonly studied Wikipedia versions in included publications (only versions with more than 5 publications are shown).](https://kindhearted-porcupine-678.convex.cloud/api/storage/63724610-c79f-4eec-a2b1-a85a33b541e0)



### 4.1.2 Multi-Language Assessment. As explained in Section 1, article quality varies significantly across different Wikipedia versions, so we tried to understand to what extent authors have studied quality assessment in multiple languages. Figure 2 shows that authors mostly focus on the English Wikipedia, but there are still some publications that consider other languages, occasionally within a machine learning context. We also discovered that 35 papers exclusively consider non-English Wikipedias [13, 15, 20, 26, 36, 39–41, 59, 60, 66, 71, 85, 89, 95, 122–124, 129, 133, 134, 142–148, 152, 174, 175, 177, 180, 181, 189].

We also analyzed how frequently authors study multiple versions at the same time. Figure 3 shows that papers almost never evaluate the quality of more than one language, but one of them [53] does a great job exploring this topic, designing different quality models for ten Wikipedia versions.

![Fig. 3. Number of assessed Wikipedia Versions per publication.](https://kindhearted-porcupine-678.convex.cloud/api/storage/fda251a5-a083-49e4-878b-c2eb83f7e5f6)

![Fig. 4. Included papers by year of publication.](https://kindhearted-porcupine-678.convex.cloud/api/storage/7defc4ba-c653-4fd6-81b0-5a4c4a6ddbf0)

## 4.2 Year of Publication

We analyzed the publication year of our results to study trends in this topic. Stvilia et al. [137, 138] studies, from 2005, were the oldest of the 149, after which interest started to grow steadily. Figure 4 shows that CL methods remained common through the years, but deep learning is clearly becoming a more prevalent approach, while metric-based studies are becoming more scarce.

## 4.3 Publication Venues

Most of the analyzed publications were published at international conferences, but we still counted many journal papers. To obtain a better overview of which publication venues are more frequent, we aggregated that information in Table 6, which shows the conferences and journals that published more than one of the papers we included in the review.

![Table 6. Overview of the Venues](https://kindhearted-porcupine-678.convex.cloud/api/storage/88cf7f18-7bf0-434e-80bc-ecb36d5873eb)

Notably, OpenSym (formerly WikiSym) is the venue that has the most publications related to our research topic. That observation is not surprising considering their significant dedication to open collaboration research. Similarly, JASIST is the peer-reviewed journal that publishes most articles on this topic.

## 4.4 Publication Influence—References and Citations

We examined and compared the number of citations and references of each included record, aiming to discover which papers were the most influential and which ones cover more sources. As shown in Figure 5, citation count varies significantly across the literature, but there are still many highly cited papers. The reference count is more stable, generally between 15 and 40. For legibility purposes, we excluded outliers (fliers) from the box plot, but we still find their analysis relevant. We found several highly cited papers, such as Stvilia et al.’s [136, 138], Wilkinson and Huberman’s [169], Blumenstock’s [14], and Hu et al.’s [62], all of which collect over 300 citations each. The publication with the most references is Halfaker and Geiger’s [53], referencing 113 other papers.

We can obtain additional conclusions from Table 7, which shows that deep learning methods are not as influential as the others. However, as we have seen, these solutions are just starting to emerge, so it is possible this observation changes in the future.

## 4.5 Abstract and Keyword Analysis

We also analyzed the most common terms in the abstract and keywords of the included publications. To do this, we first performed text normalization, which included tokenization, conversion to lowercase, removal of stop words and punctuation, and simplification of all words to their singular form. Next, for each group (abstract or keywords), we computed two measures: (1) the number

![Fig. 5.  Overview of citation (obtained from Google Scholar in March of 2023) and reference count of included publications.](https://kindhearted-porcupine-678.convex.cloud/api/storage/12d44563-aacc-46bc-873b-7558f359c26d)


![Table 7.  Top 15 Most Impactful Publications](https://kindhearted-porcupine-678.convex.cloud/api/storage/5974d4bf-b5b7-4c55-b1bc-9d3fe1b17d77)


of times each term appears in the collection of abstracts or keywords; (2) the number of abstracts or keywords in which each term appears in. These concepts are, respectively, the collection frequency ($cf_t$) and document frequency ($df_t$) as coined in the Information Retrieval area [102]. Table 8 summarizes this information, but most results are unsurprising. Aside from the obvious terms (e.g., quality, Wikipedia, article), we can see that terms related to machine learning, edits, and network analysis frequently occur.


## 4.6  Authors and Affiliations


We also decided to measure author presence across this research topic to determine which researchers study this subject more often. Table 9 summarizes this information, displaying all the authors from which we collected four or more publications, sorted by their influence, which is measured by averaging the number of citations per year of each paper we included.


![Table 8.  Term Analysis: Ten Highest Collection and Document Frequency of Abstract ($cf_{ta}, df_{ta}$) and Keywords ($cf_{tk}, df_{tk}$) Terms](https://kindhearted-porcupine-678.convex.cloud/api/storage/038f6bbd-3ab9-4688-b324-8e51ef27b809)


![Table 9.  Authors with Five or More Included Publications in the Literature Review](https://kindhearted-porcupine-678.convex.cloud/api/storage/d73d9a19-ca5d-4164-95e5-3362bec32861)


## 4.7  Datasets, Source Code, and External Tools


Regardless of the followed approach, authors generally create their datasets from Wikimedia dumps, selecting a subset of articles with varying quality distributions. Unfortunately, only 18 papers publish the datasets they use [11, 18, 23, 25, 28, 41, 43, 52, 62, 67, 79, 81, 111, 113, 128, 130, 187, 189], and most of the ones we encountered were inaccessible. In terms of implementation details, 10 papers provide the source code of their study [9, 18, 30, 31, 53, 70, 126, 131, 132, 189].


We also analyzed the used datasets to understand how they differ between studies. Figure 6 shows that machine learning datasets usually do not reach sizes as large as metric-based and other methods (e.g., feature-quality correlation approaches) do. This observation makes sense: training models are computationally expensive, so studies that assess Wikipedia quality without artificial intelligence can afford to use larger datasets.


Finally, we analyzed the tools and libraries that authors most use in their studies. Table 10 shows some of the ones we collected when analyzing the manuscripts, which we hope will be useful for helping future researchers choose between technologies.


![Fig. 6.  Size of used datasets per method type.](https://kindhearted-porcupine-678.convex.cloud/api/storage/da0cd4b8-f98d-4e41-8efe-5417355bd1ac)


![Table 10.  Relevant Tools and Libraries](https://kindhearted-porcupine-678.convex.cloud/api/storage/1d806119-37c8-4739-b3c5-54ebff29a39b)


# 5  MACHINE LEARNING APPROACHES


Here, we describe the approaches used by the 81 papers using machine learning to evaluate the quality of Wikipedia articles, comparing their performance. Authors do not always report their results using the same performance metrics, so it is not trivial to compare them directly. We wish to summarize the literature concisely but rigorously, so we will present this section’s results sorted by performance value, clearly indicating the metric chosen by each study. We will not list here any of the 13 papers that do not use Accuracy, ROC AUC, or F1-score, but we still collected those experiments for our dataset. Results here will prioritize showing AUC over accuracy, as it is often considered a better measure [96], and we will only show the F1-score if none of the two previous measures were found.


As Figure 7 suggests, 2-class and 6-class setups are much more common than the rest, so we will mainly focus on the performance of those solutions. We will separate those to allow us to better compare study performances, but we must first pay attention to the quality labels used for each class. Most 6-class studies consider Wikipedia’s Stub to FA scale [166] (usually excluding A-tier), and 2-class typically follow a Featured Article vs. Random Article approach, so those comparisons

![Fig. 7. Number of classes considered in machine learning experiments.](https://kindhearted-porcupine-678.convex.cloud/api/storage/4b50fcbd-0560-4eb2-b05b-68334ce2d8db)


should be safe. Also, since performance varies with both the number and distribution of considered classes, for each study we also show the dataset’s imbalance ratio ($IR = \# samples in the majority class / \# samples in the minority class$) [188].


Due to the nature of Machine Learning algorithms, it is unlikely that the best approach will be the same for every dataset. In fact, the No Free Lunch Theorem [170] states that all optimization algorithms have the same performance when averaged across all possible problems. Regardless, we collected all the results to understand better which algorithms were experimented with, and how performant they are in the given conditions, providing a baseline for future studies.


## 5.1 Classical Learning


We have seen that CL algorithms are the most common methods in this review: 65 publications opt to use them [4–6, 11–14, 16, 19, 21–25, 27, 28, 34, 41–44, 46, 47, 49, 52, 53, 85, 87, 89, 91, 97, 100, 101, 107, 111, 113, 115, 116, 120, 122–124, 128, 129, 133, 138–141, 148, 152–154, 157, 159–161, 171, 172, 176, 177, 179–181, 186]. Tables 11 and 12 show that decision trees, random forests, and SVMs are frequently great classical approaches, but the used performance metrics and class distribution vary so much that it is difficult to determine which solution is best. We also noticed that the best methods are almost always trained on English data, and those that are trained on multiple languages typically show much worse results on non-English data (e.g., Halfaker and Geiger [53]), which suggests there is a need for more work on multilingual assessment.


Although quality might seem a continuous measure, almost all authors decided to solve a classification task. However, wrong predictions are typically not far from the correct ones. In fact, papers sometimes present off-by-one-class accuracy in their results [52], which tend to be much higher. Only eight studies [21–25, 27, 101, 129] tackle this problem as a regression task, but the method of solving it is very similar to others, typically using a feature-based approach.


## 5.2 Deep Learning


Although less common, DL methods have recently been gaining more relevance in this field. Among the 149 publications we collected, 20 of them use deep learning [3, 9, 29–31, 50, 61, 92, 103, 111, 126, 128, 130–132, 157–159, 173, 187]. Tables 13 and 14 suggest that LSTMs and GRUs lead to the most promising results, often better than classical methods. Unfortunately, we could not collect class distribution information from many studies, which makes us uncertain about how to best assess these results. Once again, we notice a strong preference for English datasets over non-English ones.


![Table 11. Classical Learning Accuracy of 6-Class Approaches](https://kindhearted-porcupine-678.convex.cloud/api/storage/ffddedca-c698-440c-9c9b-8dc5a06d8650)


![Table 12. Classical Learning Accuracy of 2-Class Approaches (Top 10)](https://kindhearted-porcupine-678.convex.cloud/api/storage/ad803f3e-fbfd-4870-ba53-df103d70ef62)


![Table 13. Deep Learning Accuracy of 6-Class Approaches](https://kindhearted-porcupine-678.convex.cloud/api/storage/f2b4cc49-75a2-435b-9f0a-f45b3f1374e5)


![Table 14. Deep Learning Accuracy of 2-Class Approaches](https://kindhearted-porcupine-678.convex.cloud/api/storage/65e3b359-8cb0-4e44-9870-12235c78c191)


# 6 ARTICLE FEATURES AND QUALITY METRICS


Some studies, typically deep learning ones, simply feed the article’s full text to their model to obtain a quality prediction [29–31, 50, 61, 92, 126, 131, 132, 157], usually based on the Doc2Vec model [78]. However, most approaches still use article features or metrics, with and without machine learning.


This section overviews the article features and metrics we identified in this literature review. The distinction between features and metrics varies within the papers, sometimes used interchangeably. Here, we consider something a metric if it is not reasonably simple to compute and is used by the authors as a direct measure of quality (e.g., PeerReview [62]). In contrast, features are more straightforward and indirect quality measures (e.g., Character Count).


## 6.1 Article Features


We assigned a unique ID to all the features we collected from the reviewed publications, and each falls within one of the following categories:


— **Content features**, which relate to the length and structure of the article, taking into account factors such as the number of words, sections, or images.


— **Style features**, that measure how the authors write the articles, how long their phrases are, and what classes of words they use.


— **Readability features** estimate “the age or US grade level necessary to comprehend a text. (...) good articles should be well written, understandable, and free of unnecessary complexity” [21], by measuring the sentence and word complexity. They are characterized by their use of straightforward formulas that combine other types of features.


— **History features**, which analyze the review history of an article and related factors, namely the article’s age and the number of contributions.


— **Network features** are a bit more complex, as they take into account the connections between Wikipedia articles to measure their influence.


— **Popularity features** track the engagement of the page, analyzing values related to the number of views and visitors.


We based this categorization on the work of previous authors (e.g., Bassani and Viviani [12], Dalip et al. [22]), but there may be slight modifications. For instance, we consider internal link counts as content features, as we believe that any measure that can be directly computed through an article’s wikitext should belong to the Content, Style, or Readability category. Besides, although it is frequent for authors to assign internal link counts to the network category, external link counts are rarely considered network features, and we preferred to maintain consistency. Additionally, authors usually consider Content, Style, and Readability to be subcategories of *Text Features*. However, we distinguish them as different types in this review, aiming to reduce the disparity between the number of features per category.


Besides assigning a category, we also classify article features into two extra dimensions: **actionable** and **multilingual**. A feature is actionable if it can directly suggest how to improve the

![Fig. 8.  Collected features: Count per category.](https://kindhearted-porcupine-678.convex.cloud/api/storage/8b37246c-0c42-4196-9f53-fc9fcdfbc24e)


quality of the respective article, as proposed by Warncke-Wang et al. For instance, a low character count may indicate that expanding the article is beneficial for its quality. As for features that are technically manipulable but not in a relevant manner to the overall goal (e.g., revision count), we do not consider them actionable. The multilingual dimension answers the question: “Can this feature be applied to All, Most, or Some Wikipedia languages?” This is a relevant aspect when assessing, for example, readability features, whose formulas are often designed specifically for the English language. The process of evaluating these two dimensions was conducted by the authors of this study independently, and discrepancies between assessments were later discussed until an agreement was reached.


Overall, we collected 321 distinct features throughout the 149 analyzed articles. Figure 8 better displays the proportion of features per category and how they correlate to the other dimensions too.


In this sub-section, we will overview every feature category, listing the 25% most used features from each one (but never less than 15). We finish this sub-section by summarizing our findings.


### 6.1.1 Content Features


Intuitively, there is a correlation between article length and quality. Good articles should not be too long and complex, overwhelming the reader, but not too simple either, as that could signify incomplete information. Also, according to Wikipedia, stub articles (drafts) are “usually very short”, which also demonstrates that correlation.


Features related to the article structure are also essential to represent quality. Well-written articles should have a clear organization, with a balanced division of the content in sections and paragraphs. Images improve the reading experience, and references tend to increase the credibility of an article, so they are both decent indicators of quality.


### 6.1.2 Style Features


![Table 15.  List of 25% Most Used Content Features Mentioned in the Assessed Publications (Complete Feature Set and Paper Citations in Our Dataset, File ‘Full Feature List.pdf’)](https://kindhearted-porcupine-678.convex.cloud/api/storage/e93e145e-cdf9-4088-bb62-7ff42e1d472e)


### 6.1.3 Readability Features


**Automated Readability Index:** Estimates readability by combining the average word length with the average sentence size.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/d229f646-1d4a-4555-a4a8-403e3cc0a317)


**Coleman-Liau:** Similarly to ARI, estimates readability by combining the average word length with the average sentence size.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/842f91e2-cab8-4682-a7a6-3da124a15ddf)


**Difficult Word Score (DWS):** The DWS is calculated by counting the number of difficult words, which is a definition that varies between papers. According to Dang & Ignat, for example, “A word is considered difficult if it does not appear in a list of 3000 common English words that groups of fourth-grade American students could reliably understand.”


![Table 16.  List of 25% Most Used Style Features Mentioned in the Assessed Publications (Complete Feature Set and Paper Citations in Our Dataset, File ‘Full Feature List.pdf’)](https://kindhearted-porcupine-678.convex.cloud/api/storage/08e00885-db90-4971-a7a4-f97e26790494)


**Dale-Chall:** Also uses the concept of difficult words, combining it with the average sentence size to estimate readability.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/d95d6c4f-a7c6-4296-a924-83e2bac39869)


**Flesch Reading Ease:** Using the average sentence size and amount of syllables per word, computes a value between 0 and 100, where 0 indicates the text is difficult to understand.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/83ff49c8-67a3-42e7-9579-a5aff0dfd5eb)


**Flesch-Kincaid Score:** Same as FRE, but provides US grade levels instead of values between 0 and 100.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/e8ee425f-37d4-480a-ad9e-10565df783d9)


**FORCAST Readability Formula:** Measures grade level from the number of monosyllabic words in a text sample of 150 words.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/dcb5f122-d127-43d5-aa58-1ac25c8fe490)

![Table 17. List of All Readability Features Mentioned in the Assessed Publications (Complete Feature Set and Paper Citations in Our Dataset, File ‘Full Feature List.pdf’)](https://kindhearted-porcupine-678.convex.cloud/api/storage/c5f6dd96-30a6-4874-a9ca-b6ed1a13f7b2)


**Gunning Fog Index:** Uses the concept of *complexwords*, which is the number of words with three or more syllables. The higher its value, the more difficult is the text to read.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/83681f6a-fd0e-408f-bc90-dcf32eac5af6)


**Lasbarhets Index (LIX):** Very similar to *GFI*. In this case, *complexwords*, is the number of words with more than six characters. The higher its value, the more difficult is the text to read.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/ae0279a2-cbd9-4d0b-867e-193286220f43)


**Linsear Write Formula:** Let $n_1$ be the number of words with two syllables or less, and $n_2$ be the number of words with three syllables or more.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/83676423-b32b-41a4-9d4f-1f44d7d7d0e4)


**Miyazaki Readability Score:** Outputs a result between 0 and 100. The higher its value, the more difficult is the text to read.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/fc6f7ac9-1676-46b2-8278-7e94b0ee973a)


**Smog-Grading:** *polysyllables* is the average number of polysyllabic words per 30 sentences (excluding proper names). They are usually calculated from a sample of 30 sentences.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/bc6833c3-5c64-4355-87fc-1ccb39d936e8)


**Wiener Sachtextformel:** The authors propose multiple formulas, but always aim to measure the grade level required to understand a German text [10].


![Table 18. List of 15 Most Used History Features Mentioned in the Assessed Publications (Complete Feature Set and Paper Citations in Our Dataset, File ‘Full Feature List.pdf’)](https://kindhearted-porcupine-678.convex.cloud/api/storage/339f6807-0b31-4d94-aaa6-a6b04d34ed64)


### 6.1.4 History Features


### 6.1.5 Network Features


### 6.1.6 Popularity Features


### 6.1.7 Summary


We can also note a correlation between a feature’s category and its suitability for multilingual or actionable models. With a few exceptions, that pattern generally falls into what is listed in Table 21.


Finally, we decided to assess the contexts in which the collected features are used to create machine learning models. Figure 9 overviews the typical feature sets in those solutions, showing which categories are more common, and whether papers tend to use actionable and multilingual features. Content features are clearly the most prevalent, but otherwise, there is no strong preference for a specific category. We do see a preference for multilingual and actionable features in these models, which indicates that the existing literature may be useful to authors who wish to further explore these topics.


![Table 19. List of 15 Most Used Network Features Mentioned in the Assessed Publications (Complete Feature Set and Paper Citations in Our Dataset, File ‘Full Feature List.pdf’)](https://kindhearted-porcupine-678.convex.cloud/api/storage/392e16f6-c6b1-4325-a0ee-86d2484fb992)


![Table 20. List of All Popularity Features Mentioned in the Assessed Publications (Complete Feature Set and Paper Citations in Our Dataset, File ‘Full Feature List.pdf’)](https://kindhearted-porcupine-678.convex.cloud/api/storage/f6fbc96f-1519-4539-aa62-e12272c93b50)


![Table 21. Feature Category vs Actionable and Multilingual Properties](https://kindhearted-porcupine-678.convex.cloud/api/storage/ec55b0d2-655c-476e-b7d8-73e98013b8ae)

![Fig. 9. Feature characteristics within machine learning approaches (Blue—Feature Category, Orange—Actionable, Green—Multilingual). The y-axis shows the percentage of the respective features among all features used in the paper.](https://kindhearted-porcupine-678.convex.cloud/api/storage/d1e4bb3a-0bee-4497-995a-ce3ca9778b3d)


## 6.2 Quality Metrics


Sometimes authors define new metrics, using them as direct measures of quality or as input for machine learning algorithms. We found that 31 of 149 included results use a direct metric-based approach, but 71 use metrics in their study, in some way. This section describes the two most common types of metrics within the literature: Stvilia’s IQ metrics, and Text Survival metrics.


### 6.2.1 Stvilia’s IQ Metrics.


Stvilia et al. [136–138] propose 7 IQ metrics which, combining different article features, aim at evaluating Wikipedia quality more systematically. We took the definitions for each metric directly from their study [136] and define their formulas in Equations (12) to (18).


**Authority:** Authority is defined as “the degree of the reputation of an information object in a given community”.

*Connectivity* corresponds to the number of articles with at least one contributor in common with the assessed article.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/3cbced03-6c11-4156-a9b7-e968d2953f05)


**Completeness:** Authors define Completeness as “the granularity or precision of an information object’s model or content values”.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/16d4e3c6-f878-4da6-91a6-5b2e345317b5)


**Complexity:** Complexity is defined as “the degree of cognitive complexity of an information object relative to a particular activity”.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/8881bd96-b4f8-491d-a088-da39b94ada61)


**Informativeness:** Measures the amount of information in a document. *InfoNoise* represents the ratio between the size of the information and the article, measuring the amount of *noise* in the document. *Diversity* refers to the ratio between editors and total edits.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/ba8a59ae-65d7-4f66-a0c5-eabe21fe1618)


**Consistency:** Consistency is defined as “the extent to which similar attributes or elements of an information object are consistently represented with the same structure, format, and precision”.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/9135c869-62e4-49be-886e-68c0fd4ab6ee)


**Currency:** Currency corresponds to “the age of an information object” in days.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/fb55f7b2-5c32-4d66-ac8e-dccc3911570b)


**Volatility:** Volatility measures “the amount of time the information remains valid”.


![](https://kindhearted-porcupine-678.convex.cloud/api/storage/3c29051c-e1c5-40d0-abb4-40307fc95fcc)


Multiple authors talk about and experiment with these metrics [17, 18, 79, 140, 161], although rarely within a Machine Learning context. The application of Stvilia’s metrics to training ML algorithms could be worthy of experimentation.


### 6.2.2 Text Survival.


Most metric-based approaches rely on the idea of *text survival*: If a piece of the article survives many revisions, that part is likely of good quality. The most common examples are *ProbReview* and *PeerReview*, initially proposed by Hu et al. [62, 63]. Many authors use *ProbReview* [12, 21–25, 101, 158, 159] and *PeerReview* [33, 146, 159] in their works, but *text survival* is a frequent concept in metric-based approaches. For example, the proposed measures of *Transient Contribution* and *Persistent Contribution* [160, 175], *Word Persistence* [54], among others [34, 114, 142, 143, 145, 147] also rely on that notion.


# 7 DISCUSSION


This section discusses the results we obtained while attempting to answer the research questions we stated in Section 3.


## 7.1 RQ1. What are the most commonly used methods for the automatic quality assessment of Wikipedia articles?


We can identify three typical quality prediction strategies: metric-based approaches, classical machine learning models trained with article features, and deep learning methods using full text or features. Papers with other focus occasionally show up too. For example, Shen et al. [131, 132] proposed a multimodal classifier that uses both article features and a visual rendering of the document as input for quality prediction. There are also studies about quality flaw analysis and prediction [4–6, 41, 44, 92, 111, 152, 157], which identify frequent patterns of improvement.


We would also like to highlight Halfaker and Geiger’s study [53], where they propose ORES: an API-based service that supports real-time scoring of Wikipedia edits, supporting many languages, and achieving exceptional results. The study has had a great impact in this research area and the service is currently provided by Wikimedia, setting a great benchmark for all future work.


It is challenging to determine the best strategies, as each one has its advantages and drawbacks. Metric-based approaches do not require model training, and some deep learning solutions are difficult to apply in a real-time scenario (e.g., Dang & Ignat’s [31]). Regardless, our study showed that every strategy can perform effectively with the proper configuration (6-class: >60% accuracy, 2-class: >95% accuracy), and it is up to the researchers/developers to determine which solution makes the most sense for their context.


## 7.2 RQ2. How can machine learning be best applied to predict article quality, and how do different approaches compare?


Authors experiment with many machine learning algorithms, totaling 215 distinct experiments. It is not trivial to decide which algorithm is the most effective, as that is essentially dependent on the dataset definitions, but we see great performances from solutions using LSTMs, GRUs, Random Forests, and SVMs. Furthermore, boosting strategies, which combine multiple weak models into stronger ones [127], also tend to be very effective with classical algorithms (e.g., Decision Trees).


We have shown that is most studies formulate this problem as a classification task, but we would like to note Teblunthuis’s work [150], which shows that the English Wikipedia’s quality levels (FA’s, GA’s, Stubs, etc.) are not evenly distributed on a linear scale, proposing a spacing that more effectively represents how distant are the multiple levels of quality. We did not include the study in this review, because it does not directly assess the quality of Wikipedia articles, but is still an insightful analysis of Wikipedia’s quality scale.


Deep Learning is a topic that has gained significant popularity during this decade [183]. Most solutions we reviewed use classical approaches (e.g., decision trees, SVM), but we also discovered multiple deep learning solutions (e.g., LSTM). Additionally, we noticed that deep learning approaches were almost nonexistent ten years ago, while studies using classical methods for the automatic assessment of Wikipedia are not as common these days, relatively speaking.


Overall, although classical statistical learning approaches (e.g., decision trees, SVM) are more common, there has been a notable recent preference for deep learning (e.g., LSTM). Additionally, we noticed that deep learning approaches were almost nonexistent ten years ago, while studies using classical methods for the automatic assessment of Wikipedia are not as common nowadays, in relative terms. It is dangerous to directly compare results but, so far, deep learning seems to show slightly better performance than previous solutions. Furthermore, deep learning has been gaining significant popularity during the past decade [183] and, if this trend continues, it is possible that their performance eclipses the effectiveness of classical algorithms.


## 7.3 RQ3. What are the most common article features and quality metrics used to evaluate article quality in Wikipedia? How do these features compare, and how do they affect the performance of automatic assessment methods?


In this review, we collected 321 different features, each of them factoring the text of an article, its review history, how it relates to other articles within Wikipedia, or even its popularity within users.


Even though Style features are the second largest group of identified features, they are one of the least used categories. Content features, which consider only the length and structure of the article, are both the most abundant and the most frequently used ones. The other features also appear often but not as much, possibly due to their higher computation complexity.


Not all papers use simple article features to assess quality, though. Some deep learning models train with the articles’ full texts, and other studies opt for a metric-based approach, as shown in Section 5, but these approaches are not as common. Besides, although they may be used with Machine Learning, metrics are better suited for more manual approaches, so it is not surprising they do not show up as often in this review.


Some papers also compare different subsets of features regarding their effectiveness at predicting quality [21, 22, 25, 28, 44, 46, 158, 159]. By analyzing the different studies, we see that Content and Style features appear to be the most effective, but History and Network features are sometimes considered very useful for predicting quality too, so it would be wise to combine all categories.

![Fig. 10. Number of studies incorporating different methodological aspects.](https://kindhearted-porcupine-678.convex.cloud/api/storage/5a282413-cbe2-4c29-b423-d5e943baaec2)

 Readability Features also appear to generate decent results, but not so significantly as the others do, since those already combine existing features in a predefined way.


## 7.4 RQ4. Which common themes and gaps are there in the literature concerning this topic, and how can existing studies be improved to increase the adoption of automatic methods for the quality assessment of Wikipedia?


As the basis for a gap analysis, we organized multiple recurring methodology aspects into a frequency matrix, represented by the heatmap in Figure 10. Since we only pair the items two by two, this does not give us a picture of all the possible methods, but still provides quite some insight into what authors explore more and less within this field.


We can initially notice that the research within machine learning and feature-based models is extensive. There is also some work on metric-based solutions and studies focused on multiple Wikipedia languages. Nonetheless, there still exist some areas which the literature does not cover as much.


The most notable gap concerns the actionable solutions, we previously discussed. Ideally, a model would not only predict article quality but also suggest possible steps for improvement. Warncke-Wang proposes such a model, but the focus on actionable models seems to be otherwise scarce within the existing literature. Some studies propose reporting and visualization tools for analyzing the quality of Wikipedia, thus somewhat exploring this concept, but there is a clear lack of studies concerned about the topic. This review distinguishes actionable and non-actionable features in Section 6, aiming to guide authors in studying tools that assist Wikipedia readers and editors.


There is also very little work on multilingual solutions using machine learning, and none of those experiments with regression models. In addition, in studies that do design multilingual solutions, non-English model performance rarely comes close to the English one. For instance, the ORES study predicts the quality of English articles with an accuracy of 62.9% but the French model’s performance is only 44.2%. Wikipedia has millions of visitors using hundreds of versions, and as we discussed in Section 6, not every concept makes sense in every language, so it is crucial to study approaches that perform well in multiple languages.


There are some other issues not represented in the heatmap. For instance, there could also be a need for the existence of quality models specialized for specific Wikipedia categories. A couple of studies conduct several experiments with a few categories (e.g., history, biology, health), and some actually make models directed for specific fields, but there is little work on models specializing on more than one area simultaneously.


Finally, we were surprised not to see a lot of concern for the reproducibility and replicability of the studies. Only ten papers share the source code of their work (from what we discovered, at least), and shared datasets are often inaccessible. We also sometimes struggled to locate even a description of the class distribution of the datasets, which is essential when comparing machine learning results. Hopefully, future authors will give increased importance to making their work more reproducible.


In summary, the quality assessment of Wikipedia is a significantly researched topic, for which there are many diverse methods to approach it. The results we collected will help any future work related to this field, and further experimentation may help develop better quality predictors.


# 8 CONCLUSIONS AND FUTURE WORK


This study reviewed literature related to the automatic assessment of the quality of Wikipedia articles, performing an in-depth analysis of 149 different papers out of thousands of inspected results. Our findings indicate that research on this topic has fluctuated for the past few years but only started getting attention several years after the launch of Wikipedia. There are many different proposals, but most use a feature-based traditional machine learning approach and refer to Wikipedia’s content assessment standards to measure quality. We are starting to see more focus on deep learning methods, which may soon become the definite best option for this task, but it is difficult to compare results directly, since performance metrics, number of classes, and label distribution vary from study to study.


We can identify some limitations in our study, though. The most notable is the lack of non-bibliographic sources within our selection. Although our methodology should cover most journal submissions, conference papers, and other research repositories, some relevant studies may still be missed, such as Johnson’s proposed quality model. Nevertheless, nearly all relevant publications should be accessible through standard digital libraries.


Upon reviewing so much literature about the topic, we were puzzled by the fact that automatic assessment methods are still not widely used in Wikipedia. Although it is difficult to produce a direct answer, there are multiple potential explanations:


(1) Reliability: Even though some machine learning methods show impressive performance, model accuracies are far from 100%. This should not be a major issue, though, considering that, apart from B/C-tier articles, some papers show almost perfect one-off accuracy results.


(2) Complexity: As we have discussed before, quality is an extremely intricate concept with numerous properties, and some of them are much more challenging to assess than others. For instance, distinguishing a well-structured article from a poorly-structured one is trivial, compared to detecting false statements in a paragraph. Although this study does not focus so much on the trustworthiness part of IQ, all quality properties are relevant to Wikipedia users. As such, tools that do not fully grasp the essence of IQ may not be so well-received by the community.


(3) Accessibility: As we discussed in Section 7, there are not many reporting and visualization tools available for multilingual purposes, and we have seen that models are not easily transferable to other languages. We do have ORES [53] and WikiRank [176], but ORES is an API service directed to editors, and WikiRank only provides a few actionable items for improvement. Besides, without a more direct integration with Wikipedia, it is difficult for a casual Wikipedia user to learn about those tools and know how to handle them.


(4) Self-regulation: Unlike most social media, Wikipedia has no central authority, and instead relies on collaborative moderation so, by design, it cannot have a ground-truth. This is why Wikimedia is reluctant to apply AI moderation to the website [149], and may also explain why automatic quality assessment methods are not as prevalent within Wikipedia.


We cannot solve all these impediments but we believe there is potential in combining existing approaches and making a tool accessible to every Wikipedia user, providing instantaneous feedback concerning the quality of the article. Such a project could promote the widespread usage of automatic Wikipedia quality models, and the results of this review are helpful indicators of which techniques lead to better performance. Still, future researchers must design and conduct their own set of experiments, comparing languages, features, metrics, and datasets, as model performance depends on much more than just its algorithm.


At the time of writing, OpenAI’s GPT-4 has just been released [108], and the major tech companies are racing to compete against the novel ChatGPT [112]. Right now, the previously inconceivable idea of using a chatbot to predict Wikipedia article quality, explain its reasoning, and suggest items for improvement, sounds much more feasible. This would require further research and development, and it is impossible to predict how this technology will advance in the near future, but it is evident how these tools could evolve to assist the topic of this research.


# REFERENCES


[1] ACM. 2020. Artifact Review and Badging Version 1.1. Retrieved from https://www.acm.org/publications/policies/artifact-review-and-badging-current. (2020). Accessed: 2023-04-03.


[2] B. Thomas Adler, Krishnendu Chatterjee, Luca de Alfaro, Marco Faella, Ian Pye, and Vishwanath Raman. 2008. Assigning trust to wikipedia content. In Proceedings of the 4th International Symposium on Wikis (WikiSym ’08). Association for Computing Machinery, New York, NY, Article 26, 12 pages. DOI: https://doi.org/10.1145/1822258.1822293


[3] Rakshit Agrawal and Luca deAlfaro. 2016. Predicting the quality of user contributions via LSTMs. In OpenSym ’16: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–10. DOI: https://doi.org/10.1145/2957792.2957811


[4] Maik Anderka, Benno Stein, and Nedim Lipka. 2011. Detection of text quality flaws as a one-class classification problem. In CIKM ’11: ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, New York City, 2313–2316. DOI: https://doi.org/10.1145/2063576.2063954


[5] Maik Anderka, Benno Stein, and Nedim Lipka. 2011. Towards automatic quality assurance in Wikipedia. In WWW ’11: International Conference Companion on World Wide Web. Association for Computing Machinery, New York City, 5–6. DOI: https://doi.org/10.1145/1963192.1963196


[6] Maik Anderka, Benno Stein, and Nedim Lipka. 2012. Predicting quality flaws in user-generated content: The case of wikipedia. In SIGIR ’12: International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York City, 981–990. DOI: https://doi.org/10.1145/2348283.2348413


[7] Hélder Antunes and Carla Teixeira Lopes. 2019. Analyzing the adequacy of readability indicators to a Non-English language. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 11696 LNCS (2019), 149–155. DOI: https://doi.org/10.1007/978-3-030-28577-7_10/TABLES/3


[8] Ofer Arazy and Oded Nov. 2010. Determinants of wikipedia quality: The roles of global and local contribution inequality. In CSCW ’10: Conference on Computer Supported Cooperative Work. Association for Computing Machinery, New York City, 233–236. DOI: https://doi.org/10.1145/1718918.1718963

[9] Sumit Asthana, Sabrina Tobar Thommel, Aaron L. Halfaker, and Nikola Banovic. 2021. Automatically labeling low quality content on Wikipedia by leveraging patterns in editing behaviors. Proceedings of the ACM on Human-Computer Interaction 5, CSCW2 (2021), 1–23. DOI: https://dl.acm.org/doi/10.1145/3479503

[10] Richard Bamberger and Annette T. Rabin. 1984. New approaches to readability: Austrian research. The Reading Teacher 37, 6 (1984), 512–519. Retrieved from http://www.jstor.org/stable/20198517

[11] Elias Bassani and Marco Viviani. 2019. Automatically assessing the quality of Wikipedia contents. In SAC ’19: ACM/SIGAPP Symposium on Applied Computing. Association for Computing Machinery, New York City, 804–807. DOI: https://dl.acm.org/doi/10.1145/3297280.3297357

[12] Elias Bassani and Marco Viviani. 2019. Quality of Wikipedia articles: Analyzing features and building a ground truth for supervised classification. In KDIR ’19: International Conference on Knowledge Discovery and Information Retrieval. Vienna University of Technology, Vienna, Austria, 338–346. Retrieved from https://www.scitepress.org/Link.aspx?doi=10.5220/0008149303380346

[13] Grace Gimon Betancourt, Armando Segnine, Carlos Trabuco, Amira Rezgui, and Nicolas Jullien. 2016. Mining team characteristics to predict Wikipedia article quality. In OpenSym ’16: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–9. DOI: https://dl.acm.org/doi/10.1145/2957792.2971802

[14] Joshua E. Blumenstock. 2008. Size matters: Word count as a measure of quality on wikipedia. In WWW ’08: The Web Conference. Association for Computing Machinery, New York City, 1095–1096. DOI: https://dl.acm.org/doi/10.1145/1367497.1367673

[15] Fanny Chevalier, Stéphane Huot, and Jean-Daniel Fekete. 2010. WikipediaViz: Conveying article quality for casual Wikipedia readers. In PacificVis ’10: Pacific Visualization Symposium. Institute of Electrical and Electronic Engineers, New York City, 49–56. Retrieved from https://ieeexplore.ieee.org/document/5429611/

[16] Anamika Chhabra, Shubham Srivastava, S. R. S. Iyengar, and Poonam Saini. 2021. Structural analysis of wikigraph to investigate quality grades of Wikipedia articles. In WWW ’21: The Web Conference. Association for Computing Machinery, New York City, 584–590. DOI: https://dl.acm.org/doi/10.1145/3442442.3452345

[17] Luis Couto and Carla Teixeira Lopes. 2021. Assessing the quality of health-related Wikipedia articles with generic and specific metrics. In WWW ’21: The Web Conference. Association for Computing Machinery, New York City, 640–647. DOI: https://dl.acm.org/doi/10.1145/3442442.3452355

[18] Luis Couto and Carla Teixeira Lopes. 2021. Equal opportunities in the access to quality online health information? A multi-lingual study on Wikipedia. In OpenSym ’21: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–13. DOI: https://dl.acm.org/doi/10.1145/3479986.3480000

[19] Vittoria Cozza, Marinella Petrocchi, and Angelo Spognardi. 2016. A matter of words: NLP for quality evaluation of Wikipedia medical articles. In IWCE ’16: International Conference on Web Engineering. Springer, Cham, Switzerland, 448–456. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-38791-8_31

[20] Alberto Cusinato, Vincenzo Della Mea, Francesco Di Salvatore, and Stefano Mizzaro. 2009. QuWi: Quality control in Wikipedia. In WICOW ’09: Workshop on Information Credibility on the Web. Association for Computing Machinery, New York City, 27–34. DOI: https://dl.acm.org/doi/10.1145/1526993.1527001

[21] Daniel Hasan Dalip, Marcos André Gonçalves, Marco Cristo, and Pável Calado. 2009. Automatic quality assessment of content created collaboratively by web communities: A case study of wikipedia. In JCDL ’09: ACM/IEEE Joint Conference on Digital Libraries. Association for Computing Machinery, New York City, 295–304. DOI: https://dl.acm.org/doi/10.1145/1555400.1555449

[22] Daniel Hasan Dalip, Marcos André Gonçalves, Marco Cristo, and Pável Calado. 2011. Automatic assessment of document quality in web collaborative digital libraries. Journal of Data and Information Quality 2, 3 (2011), 1–30. DOI: https://dl.acm.org/doi/10.1145/2063504.2063507

[23] Daniel Hasan Dalip, Marcos André Gonçalves, Marco Cristo, and Pável Calado. 2012. On multiview-based meta-learning for automatic quality assessment of Wiki articles. In TPDL ’12: International Conference on Theory and Practice of Digital Libraries. Springer, Berlin, 234–246. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-33290-6_26

[24] Daniel Hasan Dalip, Marcos André Gonçalves, Marco Cristo, and Pável Calado. 2016. A general multiview framework for assessing the quality of collaboratively created content on web 2.0. Journal of the Association for Information Science and Technology 68, 2 (2016), 286–308. Retrieved from https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.23650

[25] Daniel Hasan Dalip, Harlley Lima, Marcos André Gonçalves, Marco Cristo, and Pável Calado. 2014. Quality assessment of collaborative content with minimal information. In JCDL ’14: ACM/IEEE Joint Conference on Digital Libraries. Association for Computing Machinery, New York City, 201–210. DOI: https://dl.acm.org/doi/10.5555/2740769.2740804

[26] Daniel Hasan Dalip, Raquel Lara Santos, Diogo Rennó Rocha de Oliveira, Valéria Freitas Amaral, Marcos André Gonçalves, Raquel Oliveira Prates, Raquel C. M. Minardi, and Jussara Marques de Almeida. 2011. GreenWiki: A tool to support users’ assessment of the quality of Wikipedia articles. In JCDL ’11: ACM/IEEE Joint Conference on Digital Libraries. Association for Computing Machinery, New York City, 469–470. DOI: https://dl.acm.org/doi/10.1145/1998076.1998190

[27] Quang-Vinh Dang. 2021. Assessing the quality of Wikipedia articles. In ICMLSC ’21: International Conference on Machine Learning and Soft Computing. Association for Computing Machinery, New York City, 1–4. DOI: https://dl.acm.org/doi/10.1145/3453800.3453801

[28] Quang-Vinh Dang and Claudia-Lavinia Ignat. 2016. Measuring quality of collaboratively edited documents: The case of Wikipedia. In CIC ’16: IEEE 2nd International Conference on Collaboration and Internet Computing. Institute of Electrical and Electronic Engineers, New York City, 266–275. Retrieved from https://ieeexplore.ieee.org/document/7809715

[29] Quang-Vinh Dang and Claudia-Lavinia Ignat. 2016. Quality Assessment of Wikipedia Articles: A Deep Learning Approach. (2016). Retrieved from https://dl.acm.org/doi/10.1145/2996442.2996447

[30] Quang-Vinh Dang and Claudia-Lavinia Ignat. 2016. Quality assessment of Wikipedia articles without feature engineering. In JCDL ’16: ACM/IEEE Joint Conference on Digital Libraries. Association for Computing Machinery, New York City, 27–30. DOI: https://dl.acm.org/doi/10.1145/2910896.2910917

[31] Quang-Vinh Dang and Claudia-Lavinia Ignat. 2017. An end-to-end learning solution for assessing the quality of Wikipedia articles. In OpenSym ’17: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–10. DOI: https://dl.acm.org/doi/10.1145/3125433.3125448

[32] Paramita Das, Bhanu Prakash Reddy Guda, Sasi Bhusan Seelaboyina, Soumya Sarkar, and Animesh Mukherjee. 2021. Quality change: Norm or exception? Measurement, analysis and detection of quality change in Wikipedia. Proceedings of the ACM on Human–Computer Interaction 6, CSCW1 (2021), 1–36. DOI: https://dl.acm.org/doi/10.1145/3512959

[33] Baptiste de La Robertie, Yoann Pitarch, and Olivier Teste. 2015. Measuring article quality in Wikipedia using the collaboration network. In ASONAM ’15: IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining. Association for Computing Machinery, New York City, 464–471. DOI: https://dl.acm.org/doi/10.1145/2808797.2808895

[34] Baptiste de La Robertie, Yoann Pitarch, and Olivier Teste. 2017. Structure-Based Features for Predicting the Quality of Articles in Wikipedia. Springer. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-51049-1_6

[35] Huijing Deng, Bernadetta Tarigan, Mihai Grigore, and Juliana Sutanto. 2015. Understanding the ‘Quality Motion’ of Wikipedia articles through semantic convergence analysis. In HCIB ’15: International Conference on HCI in Business. Springer, Cham, Switzerland, 64–75. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-20895-4_7

[36] Cecilia di Sciascio, David Strohmaier, Marcelo Errecalde, and Eduardo Veas. 2017. WikiLyzer: Interactive information quality assessment in Wikipedia. In IUI ’17: International Conference on Intelligent User Interfaces. Association for Computing Machinery, New York City, 377–388. DOI: https://dl.acm.org/doi/10.1145/3025171.3025201

[37] Pierpaolo Dondio and Stephen Barrett. 2007. Computational trust in web content quality: A comparative evalutation on the Wikipedia project. Informatica 31, 2 (2007), 151–160. Retrieved from https://arrow.tudublin.ie/scschcomart/25/

[38] Pierpaolo Dondio, Stephen Barrett, Stefan Weber, and Jean Marc Seigneur. 2006. Extracting trust from domain analysis: A case study on the Wikipedia project. In Autonomic and Trusted Computing. Springer, Berlin, 362–373.

[39] Gregory Druck, Gediminas Miklau, and Andrew McCallum. 2008. Learning to Predict the Quality of Contributions to Wikipedia. (2008). Retrieved from https://maroo.cs.umass.edu/getpdf.php?id=834

[40] Fatemeh Fahimnia, Mansoureh Damerchiloo, Mohammad Khandan, and Mahshid Eltemasi. 2022. A framework for assessing the quality of Wikipedia articles: A meta-synthesis of the literature. International Journal of Information Science and Management 20, 1 (2022), 91–118. Retrieved from https://www.magiran.com/paper/2379640

[41] Edgardo Ferretti, Leticia C. Cagnina, Viviana Paiz, Sebastián Delle Donne, Rodrigo Zacagnini, and Marcelo Errecalde. 2018. Quality flaw prediction in Spanish Wikipedia: A case study with verifiability flaws. Information Processing & Management 54, 6 (2018), 1169–1181. Retrieved from https://www.sciencedirect.com/science/article/pii/S0306457317309329?via%253Dihub

[42] Edgardo Ferretti, Donato Hernandez Fusilier, Rafael Guzmán-Cabrera, Manuel Montes y Gómez, Marcelo Errecalde, and Paolo Rosso. 2012. On the use of PU learning for quality flaw prediction in wikipedia. In CLEF ’12: Conference and Labs of the Evaluation Forum. CLEF Initiative, Rome, Italy, 1178. Retrieved from https://www.researchgate.net/publication/236565329_On_the_Use_of_PU_Learning_for_Quality_Flaw_Prediction_in_Wikipedia

[43] Edgardo Ferretti, Matías Soria, Sebastián Pérez Casseignau, Lian Pohn, Guido Urquiza, Sergio Alejandro Gómez, and Marcelo Errecalde. 2017. Towards information quality assurance in Spanish: Wikipedia. Journal of Computer Science and Technology (JCS&T) 17, 1 (2017), 29–36. Retrieved from https://www.semanticscholar.org/paper/8cba1878de84959de7a5401c9181819ee9bdf205

[44] Oliver Ferschke, Iryna Gurevych, and Marc Rittberger. 2012. FlawFinder: A modular system for predicting quality flaws in Wikipedia. In CLEF ’12: Conference and Labs of the Evaluation Forum. CLEF Initiative, Rome, Italy, 1178. Retrieved from https://www.researchgate.net/publication/235982155_FlawFinder_A_Modular_System_for_Predicting_Quality_Flaws_in_Wikipedia

[45] Zeta Field. 2015. How to Write Clearly. Publications Office of the European Union, European Union. Retrieved from https://op.europa.eu/en/publication-detail/-/publication/725b7eb0-d92e-11e5-8fea-01aa75ed71a1

[46] Lucie Flekova, Oliver Ferschke, and Iryna Gurevych. 2014. What makes a good biography?: Multidimensional quality analysis based on wikipedia article feedback data. In WWW ’14: International Conference on World Wide Web. Association for Computing Machinery, New York City, 855–866. DOI: https://dl.acm.org/doi/10.1145/2566486.2567972

[47] Yasser Ganjisaffar, Sara Javanmardi, and Cristina Lopes. 2009. Review-based ranking of Wikipedia articles. In CASON ’09: International Conference on Computational Aspects of Social Networks. Institute of Electrical and Electronic Engineers, New York City, 98–104. Retrieved from https://ieeexplore.ieee.org/document/5176107/

[48] Mouzhi Ge and Włodzimierz Lewoniewski. 2020. Developing the quality model for collaborative open data. Procedia Computer Science 176 (2020), 1883–1892. Retrieved from https://www.sciencedirect.com/science/article/pii/S187705092032130X

[49] Sindhuja Gopalan, Paolo Rosso, and Sobha Lalitha Devi. 2016. Discourse connective—A marker for identifying featured articles in biological wikipedia. Research in Computing Science 117, 1 (2016), 109–119. Retrieved from https://www.researchgate.net/journal/Research-in-Computing-Science-1870-4069

[50] Bhanu Prakash Reddy Guda, Sasi Bhusan Seelaboyina, Soumya Sarkar, and Animesh Mukherjee. 2020. NwQM: A neural quality assessment framework for Wikipedia. In EMNLP ’20: Conference on Empirical Methods in Natural Language Processing. ACL Anthology, Online, 8396–8406. Retrieved from https://aclanthology.org/2020.emnlp-main.674/

[51] Neal R. Haddaway, Matthew J. Grainger, and Charles T. Gray. 2011. Citationchaser: A tool for transparent and efficient forward and backward citation chasing in systematic searching. Research Synthesis Methods 13, 4 (2011), 533–545. Retrieved from https://doi.org/10.1002/jrsm.1563

[52] Aaron L. Halfaker. 2017. Interpolating quality dynamics in Wikipedia and demonstrating the Keilana effect. In OpenSym ’17: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–9. DOI: https://dl.acm.org/doi/10.1145/3125433.3125475

[53] Aaron L. Halfaker and R. Stuart Geiger. 2020. ORES: Lowering barriers with participatory machine learning in Wikipedia. Proceedings of the ACM on Human–Computer Interaction 4, CSCW2 (2020), 1–37. Retrieved from https://dl.acm.org/doi/10.1145/3415219

[54] Aaron L. Halfaker, Aniket Kittur, Robert Kraut, and John Riedl. 2009. A jury of your peers: Quality, experience and ownership in Wikipedia. In WikiSym ’09: International Symposium on Wikis and Open Collaboration. Association for Computing Machinery, New York City, 1–10. DOI: https://dl.acm.org/doi/10.1145/1641309.1641332

[55] Rainer Hammwöhner. 2010. Interlingual Aspects Of Wikipedia’s Quality. (2010). Retrieved from https://epub.uni-regensburg.de/15572/

[56] Jingyu Han, Xiong Fu, Kejia Chen, and Chuandong Wang. 2011. Web article quality assessment in multi-dimensional space. In WAIM ’11: International Conference on Web-Age Information Management. Springer, Berlin, 214–225. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-23535-1_20

[57] Jingyu Han, Chuandong Wang, Xiong Fu, and Kejia Chen. 2011. Probabilistic quality assessment of articles based on learning editing patterns. In CSSS ’11: International Conference on Computer Science and Service System. Institute of Electrical and Electronic Engineers, New York City, 564–570. Retrieved from https://ieeexplore.ieee.org/abstract/document/5973947

[58] Jingyu Han, Chuandong Wang, and Dawei Jiang. 2011. Probabilistic quality assessment based on article’s revision history. In DEXA ’11: International Conference on Database and Expert Systems Applications. Springer, Berlin, 574–588. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-23091-2_50

[59] Raíza Hanada, Marco Cristo, and Maria da Graça Campos Pimentel. 2013. How do metrics of link analysis correlate to quality, relevance and popularity in wikipedia?. In WebMedia ’13: Brazilian Symposium on Multimedia and the Web. Association for Computing Machinery, New York City, 105–112. DOI: https://dl.acm.org/doi/10.1145/2526188.2526198

[60] Marcelo Yuji Himoro, Raíza Hanada, Marco Cristo, and Maria da Graça Campos Pimentel. 2013. An investigation of the relationship between the amount of extra-textual data and the quality of Wikipedia articles. In WebMedia ’13: Brazilian Symposium on Multimedia and the Web. Association for Computing Machinery, New York City, 333–336. DOI: https://dl.acm.org/doi/10.1145/2526188.2526218

[61] Jingrui Hou, Jiangnan Li, and Ping Wang. 2021. Measuring quality of Wikipedia articles by feature fusion-based stack learning. In ASIST ’21: Association for Information Science and Technology. Association for Information Science & Technology, Silver Spring, Maryland, 206–217. Retrieved from https://asistdl.onlinelibrary.wiley.com/doi/10.1002/pra2.449

[62] Meiqun Hu, Ee-Peng Lim, Aixin Sun, Hady Wirawan Lauw, and Ba-Quy Vuong. 2007. Measuring article quality in wikipedia: Models and evaluation. In CIKM ’07: International Conference on Information and Knowledge Management. Association for Computing Machinery, New York City, 243–252. DOI: https://dl.acm.org/doi/10.1145/1321440.1321476

[63] Meiqun Hu, Ee-Peng Lim, Aixin Sun, Hady Wirawan Lauw, and Ba-Quy Vuong. 2007. On improving wikipedia search using article quality. In WIDM ’07: ACM International Workshop on Web Information and Data Management. Association for Computing Machinery, New York City, 145–152. DOI: https://dl.acm.org/doi/10.1145/1316902.1316926

[64] Xiao Hu, Tzi-Dong Jeremy Ng, Lu Tian, and Chi-Un Lei. 2016. Automating assessment of collaborative writing quality in multiple stages: The case of wiki. In LAK ’16: International Conference on Learning Analytics & Knowledge. Association for Computing Machinery, New York City, 518–519. DOI: https://dl.acm.org/doi/abs/10.1145/2883851.2883963

[65] Christoph Hube and Besnik Fetahu. 2018. Detecting biased statements in Wikipedia. In Companion Proceedings of the The Web Conference 2018 (WWW ’18). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 1779–1786. DOI: https://doi.org/10.1145/3184558.3191640

[66] Myshkin Ingawale, Amitava Dutta, Rahul Roy, and Priya Seetharaman. 2013. Network analysis of user generated content quality in Wikipedia. Online Information Review 37, 4 (2013), 602–619. Retrieved from https://www.emerald.com/insight/content/doi/10.1108/OIR-03-2011-0182/full/html

[67] Sara Javanmardi and Cristina Lopes. 2010. Statistical measure of quality in Wikipedia. In SOMA ’10: Workshop on Social Media Analytics. Association for Computing Machinery, New York City, 132–138. DOI: https://dl.acm.org/doi/10.1145/1964858.1964876

[68] Dariusz Jemielniak and Maciej Wilamowski. 2017. Cultural diversity of quality of information on Wikipedias. Journal of the Association for Information Science and Technology 68, 10 (2017), 2460–2470. Retrieved from https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.23901

[69] Isaac Johnson. 2022. Language-agnostic Wikipedia Article Quality Model Card. Retrieved from https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Language-agnostic_Wikipedia_article_quality_model_card. Accessed: 2023-04-04.

[70] Arash Joorabchi, Calibhe Doherty, and Jennifer Dawson. 2019. ‘WP2Cochrane’, a tool linking Wikipedia to the Cochrane Library: Results of a bibliometric analysis evaluating article quality and importance. Health Informatics Journal 26, 3 (2019), 1881–1897. Retrieved from https://journals.sagepub.com/doi/10.1177/1460458219892711

[71] Nina Khairova, Włodzimierz Lewoniewski, and Krzysztof Węcel. 2017. Estimating the quality of articles in Russian Wikipedia using the logical-linguistic model of fact extraction. In BIS ’17: International Conference on Business Information Systems. Springer, Cham, Switzerland, 28–40. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-59336-4_3

[72] Imran Khan, Shahid Hussain, Hina Gul, Muhammad Shahid, and Muhammad Jamal. 2019. An empirical study to predict the quality of Wikipedia articles. In WorldCIST ’19: World Conference on Information Systems and Technologies. Springer, Cham, Switzerland, 485–492. Retrieved from https://link.springer.com/chapter/10.1007/978-3-030-16187-3_47

[73] Khalid Al Khatib, Hinrich Schütze, and Cathleen Kantner. 2012. Automatic detection of point of view differences in Wikipedia. In Proceedings of COLING 2012. The COLING 2012 Organizing Committee, Mumbai, India, 33–50. Retrieved from https://aclanthology.org/C12-1003

[74] Aniket Kittur, Bongwon Suh, and Ed H. Chi. 2008. Can you ever trust a Wiki? Impacting perceived trustworthiness in Wikipedia. In Proceedings of the 2008 ACM Conference on Computer Supported Cooperative Work (CSCW ’08). Association for Computing Machinery, New York, NY, 477–480. DOI: https://doi.org/10.1145/1460563.1460639

[75] Rajmund Kleminski, Tomasz Kajdanowicz, Roman Bartusiak, and Przemyslaw Kazienko. 2017. On quality assesement in Wikipedia articles based on Markov random fields. In ACIIDS ’17: Asian Conference on Intelligent Information and Database Systems. Springer, Cham, Switzerland, 782–791. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-54472-4_73

[76] Andrew Kuznetsov, Margeigh Novotny, Jessica Klein, Diego Saez-Trumper, and Aniket Kittur. 2022. Templates and trust-o-meters: Towards a widely deployable indicator of trust in Wikipedia. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems (CHI ’22). Association for Computing Machinery, New York, NY, Article 125, 17 pages. Retrieved from https://doi.org/10.1145/3491102.3517523

[77] Gabriel De la Calzada and Alex Dekhtyar. 2010. On measuring the quality of Wikipedia articles. In WICOU ’10: Workshop on Information Credibility on the Web. Association for Computing Machinery, New York City, 11–18. DOI: https://dl.acm.org/doi/10.1145/1772938.1772943

[78] Quoc Le and Tomas Mikolov. 2014. Distributed representations of sentences and documents. In Proceedings of the 31st International Conference on Machine Learning (Proceedings of Machine Learning Research). Eric P. Xing and Tony Jebara (Eds.), Vol. 32. PMLR, Beijing, China, 1188–1196. Retrieved from https://proceedings.mlr.press/v32/le14.html

[79] Tao-Chi Lee and Jayakrishnan Unnikrishnan. 2013. Monitoring network structure and content quality of signal processing articles on wikipedia. In ICASSP ’13: International Conference on Acoustics. Institute of Electrical and Electronic Engineers, New York City, 8766–8770. Retrieved from https://ieeexplore.ieee.org/document/6639378

[80] Yang W. Lee, Diane M. Strong, Beverly K. Kahn, and Richard Y. Wang. 2002. AIMQ: A methodology for information quality assessment. Information & Management 40 (2002), 133–146. Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0378720602000435

[81] Jürgen Lerner and Alessandro Lomi. 2018. Knowledge categorization affects popularity and quality of Wikipedia articles. PLoS ONE 13, 1 (2018), 1–22. Retrieved from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190674

[82] Włodzimierz Lewoniewski. 2017. Enrichment of information in multilingual Wikipedia based on quality analysis. In BIS ’17: International Conference on Business Information Systems. Springer, Cham, 216–227. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-69023-0_19

[83] Włodzimierz Lewoniewski. 2018. Measures for quality assessment of articles and infoboxes in multilingual Wikipedia. In BIS ’18: International Conference on Business Information Systems. Springer, Cham, 619–633. Retrieved from https://link.springer.com/chapter/10.1007/978-3-030-04849-5_53

[84] Włodzimierz Lewoniewski, Ralf-Christian Härting, Krzysztof Węcel, Christopher Reichstein, and Witold Abramowicz. 2018. Application of SEO metrics to determine the quality of Wikipedia articles and their sources. In ICIST ’18: International Conference on Information and Software Technologies. Springer, Cham, 139–152. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-99972-2_11

[85] Włodzimierz Lewoniewski, Nina Khairova, Krzysztof Węcel, Nataliia Stratiienko, and Witold Abramowicz. 2017. Using morphological and semantic features for the quality assessment of Russian Wikipedia. In ICIST ’17: International Conference on Information and Software Technologies. Springer, Cham, 550–560. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-67642-5_46

[86] Włodzimierz Lewoniewski and Krzysztof Węcel. 2017. Relative quality assessment of Wikipedia articles in different languages using synthetic measure. In BIS ’17: International Conference on Business Information Systems. Springer, Cham, 282–292. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-69023-0_24

[87] Włodzimierz Lewoniewski, Krzysztof Węcel, and Witold Abramowicz. 2016. Quality and importance of Wikipedia articles in different languages. In ICIST ’16: International Conference on Information and Software Technologies. Springer, Cham, 613–624. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-46254-7_50

[88] Włodzimierz Lewoniewski, Krzysztof Węcel, and Witold Abramowicz. 2017. Relative quality and popularity evaluation of multilingual Wikipedia articles. Informatics 4, 4 (2017), 43. Retrieved from https://www.mdpi.com/2227-9709/4/4/43

[89] Włodzimierz Lewoniewski, Krzysztof Węcel, and Witold Abramowicz. 2018. Determining quality of articles in polish Wikipedia based on linguistic features. In ICIST ’18: International Conference on Information and Software Technologies. Springer, Cham, 546–558. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-99972-2_45

[90] Włodzimierz Lewoniewski, Krzysztof Węcel, and Witold Abramowicz. 2019. Multilingual ranking of Wikipedia articles with quality and popularity assessment in different topics. Computers 8, 3 (2019), 60. Retrieved from https://www.mdpi.com/2073-431X/8/3/60

[91] Elisabeth Lex, Michael Voelske, Marcelo Errecalde, Edgardo Ferretti, Leticia C. Cagnina, Christopher Horn, Benno Stein, and Michael Granitzer. 2012. Measuring the quality of web content using factual information. In WebQuality ’12: Joint WICOW/AIRWeb Workshop on Web Quality. Association for Computing Machinery, New York City, 7–10. DOI: https://doi.org/10.1145/2184305.2184308

[92] Muyan Li, Heshen Zhou, Jingrui Hou, Ping Wang, and Erpei Gao. 2022. Is cross-linguistic advert flaw detection in Wikipedia feasible? A multilingual-BERT-based transfer learning approach. Knowledge-Based Systems 252, 109330 (2022). Retrieved from https://www.sciencedirect.com/science/article/pii/S0950705122006670

[93] Xinyi Li, Jintao Tang, Ting Wang, Zhunchen Luo, and Maarten de Rijke. 2015. Automatically assessing Wikipedia article quality by exploiting article-editor networks. In ECIR ’15: European Conference on Information Retrieval. Springer, Cham, 574–580. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-16354-3_64

[94] Ee-Peng Lim, Ba-Quy Vuong, Hady Wirawan Lauw, and Aixin Sun. 2006. Measuring qualities of articles contributed by online communities. In WI ’16: IEEE WIC ACM International Conference on Web Intelligence. Institute of Electrical and Electronic Engineers, New York City, 81–87. Retrieved from https://ieeexplore.ieee.org/document/4061345

[95] Yan Lin and Chenxi Wang. 2020. Wisdom of crowds: The effect of participant composition and contribution behavior on Wikipedia article quality. Journal of Knowledge Management 24, 2 (2020), 324–345. Retrieved from https://www.emerald.com/insight/content/doi/10.1108/JKM-08-2019-0416/full/html

[96] Charles X. Ling, Jin Huang, and Harry Zhang. 2003. AUC: A better measure than accuracy in comparing learning algorithms. In Advances in Artificial Intelligence: 16th Conference of the Canadian Society for Computational Studies of Intelligence, AI 2003, Halifax, Canada, June 11–13, 2003, Proceedings 16. Springer, Springer, Berlin, 329–341.

[97] Nedim Lipka and Benno Stein. 2010. Identifying featured articles in wikipedia: Writing style matters. In WWW ’10: International Conference on the World Wide Web. Association for Computing Machinery, New York City, 1147–1148. DOI: https://dl.acm.org/doi/10.1145/1772690.1772847

[98] Jun Liu and Sudha Ram. 2011. Who does what: Collaboration patterns in the wikipedia and their impact on article quality. ACM Transactions on Management Information Systems 2, 2 (2011), 1–23. DOI: https://dl.acm.org/doi/10.1145/1985347.1985352

[99] Jun Liu and Sudha Ram. 2018. Using big data and network analysis to understand Wikipedia article quality. Data & Knowledge Engineering 115 (2018), 80–93. Retrieved from https://www.sciencedirect.com/science/article/pii/S0169023X18300685?via%253Dihub

[100] Yuqing Lu, Lei Zhang, and Juan-Zi Li. 2013. Evaluating article quality and editor reputation in Wikipedia. In CSWS ’13: China Semantic Web Symposium and Web Science Conference. Springer, Berlin, 215–227. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-54025-7_19

[101] Luiz Felipe Gonçalves Magalhães, Marcos André Gonçalves, Sérgio Daniel Canuto, Daniel Hasan Dalip, Marco Cristo, and Pável Calado. 2019. Quality assessment of collaboratively-created web content with no manual intervention based on soft multi-view generation. Expert Systems with Applications 132 (2019), 226–238. Retrieved from https://www.sciencedirect.com/science/article/pii/S0957417419302830

[102] C. D. Manning, P. Raghavan, and H. Schutze. 2008. Introduction to Information Retrieval. Cambridge University Press, Cambridge, England. Retrieved from https://books.google.pt/books?id=t1Posh4uwVcC

[103] Edison Marrese-Taylor, Pablo Loyola, and Yutaka Matsuo. 2019. An edit-centric approach for Wikipedia article quality assessment. In WNUT ’19: Workshop on Noisy User-generated Text. ACL Anthology, Online, 381–386. Retrieved from https://aclanthology.org/D19-5550/

[104] Emanuel Marzini, Angelo Spognardi, Ilaria Matteucci, Paolo Mori, Marinella Petrocchi, and Riccardo Conti. 2014. Improved Automatic Maturity Assessment of Wikipedia Medical Articles. In OTM ’14: Confederated International Conferences “On the Move to Meaningful Internet Systems”. Springer, Berlin, 612–662. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-45563-0_37

[105] Sai T. Moturu and Huan Liu. 2009. Evaluating the trustworthiness of Wikipedia articles through quality and credibility. In WikiSym ’09: International Symposium on Wikis and Open Collaboration. Association for Computing Machinery, New York City, 1–2. DOI: https://dl.acm.org/doi/10.1145/1641309.1641349

[106] Pedro Miguel Moás and Carla Teixeira Lopes. 2023. Automatic Quality Assessment of Wikipedia Articles—A Systematic Literature Review Dataset [Dataset]. INESC TEC. (2023). DOI: https://doi.org/10.25747/s5fa-d428

[107] Nir Ofek and Lior Rokach. 2015. A classifier to determine which Wikipedia biographies will be accepted. Journal of the Association for Information Science and Technology 66, 1 (2015), 213–218. Retrieved from https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.23199

[108] OpenAI. 2023. GPT-4 Technical Report. (2023). Retrieved from arXiv:cs.CL/2303.08774

[109] Matthew J. Page, Joanne E. McKenzie, Patrick M. Bossuyt, Isabelle Boutron, Tammy C. Hoffmann, Cynthia D. Mulrow, Larissa Shamseer, Jennifer M. Tetzlaff, Elie A. Akl, Sue E. Brennan, Roger Chou, Julie Glanville, Jeremy M. Grimshaw, Asbjørn Hróbjartsson, Manoj M. Lalu, Tianjing Li, Elizabeth W. Loder, Evan Mayo-Wilson, Steve McDonald, Luke A. McGuinness, Lesley A. Stewart, James Thomas, Andrea C. Tricco, Vivian A. Welch, Penny Whiting, and David Moher. 2021. The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. BMJ 372 (3 2021). DOI: https://doi.org/10.1136/BMJ.N71

[110] Matthew J. Page, David Moher, Patrick M. Bossuyt, Isabelle Boutron, Tammy C. Hoffmann, Cynthia D. Mulrow, Larissa Shamseer, Jennifer M. Tetzlaff, Elie A. Akl, Sue E. Brennan, Roger Chou, Julie Glanville, Jeremy M. Grimshaw, Asbjørn Hróbjartsson, Manoj M. Lalu, Tianjing Li, Elizabeth W. Loder, Evan Mayo-Wilson, Steve McDonald, Luke A. McGuinness, Lesley A. Stewart, James Thomas, Andrea C. Tricco, Vivian A. Welch, Penny Whiting, and Joanne E. McKenzie. 2021. PRISMA 2020 explanation and elaboration: Updated guidance and exemplars for reporting systematic reviews. BMJ 372 (2021), 1–36. DOI: https://doi.org/10.1136/bmj.n160 arXiv:https://www.bmj.com/content/372/bmj.n160.full.pdf

[111] Gerónimo Bazán Pereyra, Carolina Cuello, Gianfranco Capodici, Vanessa Jofré, Edgardo Ferretti, Rodolfo Bonnin, and Marcelo Errecalde. 2019. Predicting information quality flaws in Wikipedia by using classical and deep learning approaches. In CACIC ’19: Argentine Congress of Computer Science. Springer, Cham, 3–18. Retrieved from https://link.springer.com/chapter/10.1007/978-3-030-48325-8_1

[112] David Pierce. 2023. ChatGPT Started a New Kind of AI Race—and Made Text Boxes Cool Again. Retrieved from https://www.theverge.com/2023/3/26/23655456/chatgpt-bard-bing-ai-race-text-boxes. Accessed: 2023-04-01.

[113] Lian Pohn, Edgardo Ferretti, and Marcelo Errecalde. 2014. Identifying Featured Articles in Spanish Wikipedia. (2014). Retrieved from http://sedici.unlp.edu.ar/bitstream/handle/10915/42288/Documento_completo.pdf?sequence=1

[114] Xiangju Qin and Pádraig Cunningham. 2012. Assessing the Quality of Wikipedia Pages Using Edit Longevity and Contributor Centrality. (2012). Retrieved from https://arxiv.org/abs/1206.2517

[115] Narun K. Raman, Nathaniel Sauerberg, Jonah Fisher, and Sneha Narayan. 2020. Classifying Wikipedia article quality with revision history networks. In OpenSym ’20: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–7. DOI: https://dl.acm.org/doi/10.1145/3412569.3412581

[116] Laura Rassbach, Trevor Blackford, and Brian Mingus. 2007. Exploring the feasibility of automatically rating online article quality. In Wikimania ’07: Wikimania Conference. Wikimedia Foundation, San Francisco, California. Retrieved from https://scholar.google.pt/citations?view_op=view_citation%26hl=pt-PT%26user=T_sFnwoAAAAJ%26citation_for_view=T_sFnwoAAAAJ:u-x6o8ySG0sC

[117] Dwaipayan Roy, Sumit Bhatia, and Prateek Jain. 2020. A topic-aligned multilingual corpus of Wikipedia articles for studying information asymmetry in low resource languages. In Proceedings of the 12th Language Resources and Evaluation Conference. European Language Resources Association, Marseille, France, 2373–2380. Retrieved from https://aclanthology.org/2020.lrec-1.289

[118] Dwaipayan Roy, Sumit Bhatia, and Prateek Jain. 2022. Information asymmetry in Wikipedia across different languages: A statistical analysis. Journal of the Association for Information Science and Technology 73, 3 (3 2022), 347–361.

[119] Thorsten Ruprechter, Tiago Santos, and Denis Helic. 2019. On the relation of edit behavior, link structure, and article quality on Wikipedia. In COMPLEX NETWORKS ’19: International Workshop on Complex Networks & Their Applications. Springer, Cham, Switzerland, 242–254. Retrieved from https://link.springer.com/chapter/10.1007/978-3-030-36683-4_20

[120] Thorsten Ruprechter, Tiago Santos, and Denis Helic. 2020. Relating Wikipedia article quality to edit behavior and link structure. Applied Network Science 5, 61 (2020), 1–20. Retrieved from https://appliednetsci.springeropen.com/articles/10.1007/s41109-020-00305-y

[121] Giuseppe De Ruvo and Antonella Santone. 2015. Analysing wiki quality using probabilistic model checking. In WET ICE ’15: IEEE International Workshop on Enabling Technologies: Infrastructure for Collaborative Enterprises. Institute of Electrical and Electronic Engineers, New York City, 224–229. Retrieved from https://ieeexplore.ieee.org/document/71943655

[122] Kanchana Saengthongpattana and Nuanwan Soonthornphisaj. 2014. Assessing the quality of Thai Wikipedia articles using concept and statistical features. In WorldCIST ’14: World Conference on Information Systems and Technologies. Springer, Cham, Switzerland, 513–523. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-05951-8_49

[123] Kanchana Saengthongpattana, Thepchai Supnithi, and Nuanwan Soonthornphisaj. 2017. Ontology-based classifiers for Wikipedia article quality classification. In iSAI-NLP ’17: International Joint Symposium on Artificial Intelligence and Natural Language Processing. Springer, Cham, Switzerland, 68–81. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-94703-7_7

[124] Kanchana Saengthongpattana, Thepchai Supnithi, and Nuanwan Soonthornphisaj. 2018. Quality classification of ASEAN Wikipedia articles using statistical features. In iSAI-NLP ’18: International Joint Symposium on Artificial Intelligence and Natural Language Processing. Institute of Electrical and Electronic Engineers, New York City, 1–6. Retrieved from https://ieeexplore.ieee.org/document/8692954/

[125] Flavia Salutari, Diego Da Hora Gilles, Dubuc, and Dario Rossi. 2019. A large-scale study of Wikipedia users’ quality of experience. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, 3194–3200. DOI: https://doi.org/10.1145/3308558.3313467

[126] Soumya Sarkar, Bhanu Prakash Reddy Guda, Sandipan Sikdar, and Animesh Mukherjee. 2019. StRE: Self attentive edit quality prediction in Wikipedia. In ACL ’19: Annual Meeting of the Association for Computational Linguistics. ACL Anthology, Online, 3962–3972. Retrieved from https://aclanthology.org/P19-1387/

[127] Robert E. Schapire. 2003. The Boosting Approach to Machine Learning: An Overview. Springer New York, New York, NY, 149–171. DOI: https://doi.org/10.1007/978-0-387-21579-2_9

[128] Manuel Schmidt and Eva Zangerle. 2019. Article quality classification on Wikipedia: Introducing document embeddings and content features. In OpenSym ’19: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–8. DOI: https://dl.acm.org/doi/10.1145/3306446.3340831

[129] Seyedtaha Seyedsadr, Mohammadali Afsharkazemi, and Hashem Nikoomaram. 2016. Qualifying articles of persian Wikipedia encyclopedia through J48 algorithm, ANFIS and subtractive clustering. Automation 3, 6 (2016), 141–153. Retrieved from https://www.sciencepublishinggroup.com/journal/paperinfo?journalid=134%26doi=10.11648/j.acis.20150306.18

[130] Aili Shen, Jianzhong Qi, and Timothy Baldwin. 2017. A hybrid model for quality assessment of Wikipedia articles. In ALTA ’17: Australasian Language Technology Association Workshop. ACL Anthology, Online, 43–52. Retrieved from https://aclanthology.org/U17-1005/

[131] Aili Shen, Bahar Salehi, Timothy Baldwin, and Jianzhong Qi. 2019. A joint model for multimodal document quality assessment. In JCDL ’19: Joint conference on digital libraries. Association for Computing Machinery, New York City, 107–110. DOI: https://dl.acm.org/doi/10.1145/JCDL.2019.00024

[132] Aili Shen, Bahar Salehi, Jainzhong Qi, and Timothy Baldwin. 2020. A multimodal approach to assessing document quality. Journal of Artificial Intelligence Research 68 (2020), 607–632. Retrieved from https://www.jair.org/index.php/jair/article/view/11647

[133] Nuanwan Soonthornphisaj and Peerapoom Paengporn. 2017. Thai Wikipedia article quality filtering algorithm. In IMECS ’17: International MultiConference of Engineers and Computer Scientists. International Association of Engineers, Hong Kong, China. Retrieved from https://www.iaeng.org/publication/IMECS2017/IMECS2017_pp299-305.pdf

[134] Klaus Stein and Claudia Hess. 2007. Does it matter who contributes: A study on featured articles in the german wikipedia. In HT ’07: Conference on Hypertext and Hypermedia. Association for Computing Machinery, New York City, 171–174. DOI: https://dl.acm.org/doi/10.1145/1286240.1286290

[135] Besiki Stvilia, Abdullah Al-Faraj, and Yong Jeong Yi. 2009. Issues of cross-contextual information quality evaluation–The case of Arabic, English, and Korean Wikipedias. Library & Information Science Research 31, 4 (2009), 232–239. Retrieved from https://www.sciencedirect.com/science/article/pii/S0740818809000954

[136] Besiki Stvilia, Les Gasser, Michael B. Twidale, and Linda C. Smith. 2007. A framework for information quality assessment. Journal of the Association for Information Science and Technology 58, 12 (2007), 1720–1733. Retrieved from https://onlinelibrary.wiley.com/doi/10.1002/asi.20652

[137] Besiki Stvilia, Michael B. Twidale, Les Gasser, and Linda C. Smith. 2005. Information quality discussions in wikipedia. In ICKM ’05: International Conference on Knowledge Management. Universiti Putra Malaysia, Seri Kembangan, Malaysia. Retrieved from https://www.researchgate.net/publication/200773232_Information_Quality_Discussions_in_Wikipedia

[138] Besiki Stvilia, Michael B. Twidale, Linda C. Smith, and Les Gasser. 2005. Assessing information quality of a community-based encyclopedia. In ICIQ ’05: International Conference on Information Quality. Massachusetts Institute of Technology, Cambridge, Massachusetts, 442–454. Retrieved from https://www.semanticscholar.org/paper/Assessing-Information-Quality-of-a-Community-Based-Stvilia-Twidale/dd888dddccc2075a44f99ec2380fda652040afaf

[139] Qi Su and Pengyuan Liu. 2015. A psycho-lexical approach to the assessment of information quality on Wikipedia. In WI-IAT ’15: IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology. Institute of Electrical and Electronic Engineers, New York City, 184–187. Retrieved from https://ieeexplore.ieee.org/document/7397452

[140] Chinthani Sugandhika and Supunmali Ahangama. 2022. Assessing information quality of Wikipedia articles through Google’s E-A-T model. IEEE Access 10 (2022), 52196–52209. Retrieved from https://ieeexplore.ieee.org/document/9770051

[141] Chinthani Sugandhika, Supunmali Ahangama, and Sapumal Ahangama. 2021. Modelling Wikipedia’s information quality using informativeness, reliability and authority. In ICAC ’21: International Conference on Advancements in Computing. Institute of Electrical and Electronic Engineers, New York City, 169–174. Retrieved from https://ieeexplore.ieee.org/document/9671092

[142] Yu Suzuki. 2012. Assessing quality values of Wikipedia articles using implicit positive and negative ratings. In WAIM ’12: International Conference on Web-Age Information Management. Springer, Berlin, 127–138. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-32281-5_13

[143] Yu Suzuki. 2013. Effects of implicit positive ratings for quality assessment of Wikipedia articles. Journal of Information Processing 21, 2 (2013), 342–348. Retrieved from https://www.jstage.jst.go.jp/article/ipsjjip/21/2/21_342/_article

[144] Yu Suzuki. 2015. Quality assessment of Wikipedia articles using h-index. Journal of Information Processing 23, 1 (2015), 22–30. Retrieved from https://www.jstage.jst.go.jp/article/ipsjjip/23/1/23_22/_article

[145] Yu Suzuki and Masatoshi Yoshikawa. 2012. Mutual evaluation of editors and texts for assessing quality of Wikipedia articles. In WikiSym ’12: International Symposium on Wikis and Open Collaboration. Association for Computing Machinery, New York City, 1–10. DOI: https://dl.acm.org/doi/10.1145/2462932.2462956

[146] Yu Suzuki and Masatoshi Yoshikawa. 2012. QualityRank: Assessing quality of Wikipedia articles by mutually evaluating editors and texts. In HT ’12: ACM Conference on Hypertext & Social Media. Association for Computing Machinery, New York City, 307–308. DOI: https://dl.acm.org/doi/10.1145/2309996.2310047

[147] Yu Suzuki and Masatoshi Yoshikawa. 2013. Assessing quality score of Wikipedia article using mutual evaluation of editors and texts. In CIKM ’13: ACM International Conference on Information & Knowledge Management. Association for Computing Machinery, New York City, 1722–1732. DOI: https://dl.acm.org/doi/10.1145/2505515.2505610

[148] Marcin Sydow, Katarzyna Baraniak, and Paweł Teisseyre. 2017. Diversity of editors and teams versus quality of cooperative work: Experiments on wikipedia. Journal of Intelligent Information Systems 48 (2017), 601–632. Retrieved from https://link.springer.com/article/10.1007/s10844-016-0428-1

[149] Diego Sáez-Trumper. 2021. Disinformation and AI: The Differences Between Wikipedia and Social Media. Retrieved from https://diff.wikimedia.org/2021/09/15/disinformation-and-ai-the-differences-between-wikipedia-and-social-media/. Accessed: 2023-04-04.

[150] Nathan Teblunthuis. 2021. Measuring Wikipedia article quality in one dimension by extending ORES with ordinal regression. In Proceedings of the 17th International Symposium on Open Collaboration (OpenSym ’21). Association for Computing Machinery, New York, NY, Article 5, 10 pages. DOI: https://doi.org/10.1145/3479986.3479991

[151] Michail Tsikerdekis. 2017. Cumulative experience and recent behavior and their relation to content quality on Wikipedia. Interacting with Computers 29, 5 (2017), 737–754. Retrieved from https://academic.oup.com/iwc/article/29/5/737/3885842

[152] Guido Urquiza, Matías Soria, Sebastián Pérez Casseignau, Edgardo Ferretti, Sergio Alejandro Gómez, and Marcelo Errecalde. 2016. On the assessment of information quality in Spanish Wikipedia. In CACIC ’19: Argentine Congress of Computer Science. National University of La Plata, La Plata, Argentina, 702–711. Retrieved from http://sedici.unlp.edu.ar/handle/10915/56750

[153] Srikar Velichety. 2019. Quality assessment of peer-produced content in knowledge repositories using big data and social networks: The case of implicit collaboration in Wikipedia. ACM SIGMIS Database: The DATABASE for Advances in Information Systems 50, 4 (2019), 28–51. DOI: https://dl.acm.org/doi/10.1145/3371041.3371045

[154] Srikar Velichety, Sudha Ram, and Jesse Bockstedt. 2019. Quality assessment of peer-produced content in knowledge repositories using development and coordination activities. Journal of Management Information Systems 36, 2 (2019), 478–512. Retrieved from https://www.tandfonline.com/doi/full/10.1080/07421222.2019.1598692

[155] Carlos G. Velázquez, Leticia C. Cagnina, and Marcelo Errecalde. 2017. On the feasibility of external factual support as Wikipedia’s quality metric. Processamiento del Lenguaje Natural 58 (2017), 93–100. Retrieved from http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/5417

[156] Nicholas Vincent and Brent Hecht. 2021. A deeper investigation of the importance of Wikipedia links to search engine results. Proceedings of the ACM on Human–Computer Interaction 5, CSCW1 (2021), 18. DOI: https://doi.org/10.1145/3449078

[157] Ping Wang, Muyan Li, Xiaodan Li, Heshen Zhou, and Jingrui Hou. 2021. A hybrid approach to classifying Wikipedia article quality flaws with feature fusion framework. Expert Systems with Applications 181, 1 (2021), 115089. Retrieved from https://www.sciencedirect.com/science/article/pii/S0957417421005303?via%253Dihub

[158] Ping Wang and Xiaodan Li. 2020. Assessing the quality of information on wikipedia: A deep-learning approach. Journal of the Association for Information Science and Technology 71, 1 (2020), 16–28. Retrieved from https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.24210

[159] Ping Wang, Xiaodan Li, and Renli Wu. 2019. A deep learning-based quality assessment model of collaboratively edited documents: A case study of Wikipedia. Journal of Information Science 47, 2 (2019), 176–191. DOI: https://journals.sagepub.com/doi/10.1177/0165551519877646

[160] Se Wang and Mizuho Iwaihara. 2010. Quality evaluation of Wikipedia articles through edit history and editor groups. In APWeb ’11: Asia-Pacific Web Conference. Springer, Berlin, 188–199. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-20291-9_20

[161] Morten Warncke-Wang, Dan Cosley, and John Riedl. 2013. Tell me more: An actionable quality model for Wikipedia. In WikiSym ’13: International Symposium on Open Collaboration. Association for Computing Machinery, New York City, 1–10. DOI: https://dl.acm.org/doi/10.1145/2491055.2491063

[162] Wikimedia. 2022. Wikipedia Statistics—Edit and Revert Trends. Retrieved from https://stats.wikimedia.org/EN/EditsRevertsEN.htm. Accessed: 2023-04-03.

[163] Wikimedia. 2022. Wikistats - Statistics For Wikimedia Projects. Retrieved from https://stats.wikimedia.org. Accessed: 2023-04-03.

[164] Wikipedia. 2022. List of Wikipedias. Retrieved from https://meta.wikimedia.org/wiki/List_of_Wikipedias. Accessed: 2023-04-03.

[165] Wikipedia. 2022. Wikipedia. Retrieved from https://en.wikipedia.org/wiki/Wikipedia. Accessed: 2023-04-03.

[166] Wikipedia. 2022. Wikipedia: Content Assessment. Retrieved from https://en.wikipedia.org/wiki/Wikipedia:Content_assessment. Accessed: 2023-04-03.

[167] Wikipedia. 2022. Wikipedia: Size of Wikipedia. Retrieved from https://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia. Accessed: 2023-04-03.

[168] Wikipedia. 2023. Wikipedia: Manual of Style. Retrieved from https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/. Accessed: 2023-04-04.

[169] Dennis Wilkinson and Bernardo Huberman. 2007. Cooperation and quality in wikipedia. In WikiSym ’07: International Symposium on Wikis. Association for Computing Machinery, New York City, 157–164. DOI: https://dl.acm.org/doi/10.1145/1296951.1296968

[170] David H. Wolpert and William G. Macready. 1997. No free lunch theorems for optimization. IEEE Transactions on Evolutionary Computation 1, 1 (1997), 67.

[171] Guangyu Wu, Martin Harrigan, and Pádraig Cunningham. 2011. Characterizing Wikipedia pages using edit network motif profiles. In SMUC ’11: International Workshop on Search and Mining User-Generated Contents. Association for Computing Machinery, New York City, 45–52. DOI: https://dl.acm.org/doi/10.1145/2065023.2065036

[172] Guangyu Wu, Martin Harrigan, and Pádraig Cunningham. 2012. Classifying Wikipedia articles using network motif counts and radios. In WikiSym ’12: International Symposium on Wikis and Open Collaboration. Association for Computing Machinery, New York City, 1–12. DOI: https://dl.acm.org/doi/10.1145/2462932.2462948

[173] Kewen Wu, Qinghua Zhu, Yuxiang Zhao, and Hua Zheng. 2010. Mining the factors affecting the quality of Wikipedia articles. In ISME ’10: International Conference of Information Science and Management Engineering. Institute of Electrical and Electronic Engineers, New York City, 343–346. Retrieved from https://ieeexplore.ieee.org/document/5572324


[174] Thomas Wöhner, Sebastian Köhler, and Ralf Peters. 2015. Good authors = good articles? - How Wikis work. In WI ’15: International Conference on Wirtschaftsinformatik. Association for Information Systems, Atlanta, Georgia. Retrieved from https://aisel.aisnet.org/wi2015/59/?utm_source=aisel.aisnet.org%252Fwi2015%252F59%26utm_medium=PDF%26utm_campaign=PDFCoverPages


[175] Thomas Wöhner and Ralf Peters. 2009. Assessing the quality of Wikipedia articles with lifecycle based metrics. In WikiSym ’09: International Symposium on Wikis and Open Collaboration. Association for Computing Machinery, New York City, 1–10. DOI: https://dl.acm.org/doi/10.1145/1641309.1641333


[176] Krzysztof Węcel and Włodzimierz Lewoniewski. 2015. Modelling the quality of attributes in Wikipedia infoboxes. In BIS ’15: International Conference on Business Information Systems. Springer, Cham, Switzerland, 308–320. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-26762-3_27


[177] Kui Xiao, Bing Li, Peng He, and Xi hui Yang. 2013. Detection of article qualities in the Chinese Wikipedia based on C4.5 decision tree. In KSEM ’13: International Conference on Knowledge Science. Springer, Berlin, 444–452. Retrieved from https://link.springer.com/chapter/10.1007/978-3-642-39787-5_36


[178] Feiyu Xu, Hans Uszkoreit, Yangzhou Du, Wei Fan, Dongyan Zhao, and Jun Zhu. 2019. Explainable AI: A brief survey on history, research areas, approaches and challenges. In Natural Language Processing and Chinese Computing: 8th CCF International Conference, NLPCC 2019, Dunhuang, China, October 9–14, 2019, Proceedings, Part II 8. Springer International Publishing, Cham, 563–574.


[179] Yanxiang Xu and Tiejian Luo. 2011. Measuring article quality in Wikipedia: Lexical clue model. In SWS ’11: Symposium on Web Society. Institute of Electrical and Electronic Engineers, New York City, 141–146. Retrieved from https://ieeexplore.ieee.org/document/6101286


[180] Adnan Yahya, Afnan Ahmad, Alaa Assaf, Rawan Khater, and Ali Salhi. 2020. Models for Arabic document quality assessment. In BIS ’20: International Conference on Business Information Systems. Springer, Cham, Switzerland, 297–310. Retrieved from https://link.springer.com/chapter/10.1007/978-3-030-61146-0_24


[181] Adnan Yahya and Ali Salhi. 2014. Quality assessment of Arabic web content: The case of the Arabic Wikipedia. In IIT ’14: International Conference on Innovations in Information Technology. Institute of Electrical and Electronic Engineers, New York City, 36–41. Retrieved from https://ieeexplore.ieee.org/document/6987558


[182] Diyi Yang, Aaron L. Halfaker, Robert Kraut, and Eduard Hovy. 2016. Who did what: Editor role identification in Wikipedia. In ICWSM ’16: International AAAI Conference on Web and Social Media. Association for the Advancement of Artificial Intelligence, Palo Alto, California, 446–455. Retrieved from https://ojs.aaai.org/index.php/ICWSM/article/view/14732


[183] M. Mutlu Yapıcı, Adem Tekerek, and Nurettin Topaloğlu. 2019. Literature review of deep learning research areas. Gazi Mühendislik Bilimleri Dergisi 5, 3 (2019), 188–215. DOI: https://doi.org/10.30855/gmbd.2019.03.01


[184] Linfeng Yu and Mizuho Iwaihara. 2018. Finding high quality documents through link and click graphs. In IIAI-AAI ’18: International Congress on Advanced Applied Informatics. Institute of Electrical and Electronic Engineers, New York City, 49–54. Retrieved from https://ieeexplore.ieee.org/abstract/document/8693372


[185] Honglei Zeng, Maher A Alhossaini, Li Ding, Richard Fikes, and Deborah L. McGuinness. 2006. Computing Trust from Revision History. Technical Report. Stanford Univ Ca Knowledge Systems LAB. Retrieved from https://apps.dtic.mil/sti/citations/ADA454704


[186] Ning Zhang, Lingyun Ruan, and Luo Si. 2015. Predicting Low-Quality Wikipedia Articles Using User’s Judgements. Springer, Cham, Switzerland. Retrieved from https://link.springer.com/chapter/10.1007/978-3-319-05467-4_6


[187] Shiyue Zhang, Zheng Hu, Chunhong Zhang, and Ke Yu. 2018. History-based article quality assessment on Wikipedia. In BIGCOMP ’18: International Conference on Big Data and Smart Computing. Institute of Electrical and Electronic Engineers, New York City, 1–8. Retrieved from https://ieeexplore.ieee.org/document/8367090


[188] Rui Zhu, Yiwen Guo, and Jing-Hao Xue. 2020. Adjusting the imbalance ratio by the dimensionality of imbalanced data. Pattern Recognition Letters 133 (2020), 217–223. DOI: https://doi.org/10.1016/j.patrec.2020.03.004


[189] Didem Ölçer and Tuğba Taşkaya Temizel. 2022. Quality assessment of web-based information on type 2 diabetes. Online Information Review 46, 4 (2022), 715–732. Retrieved from https://www.emerald.com/insight/content/doi/10.1108/OIR-02-2021-0089/full/html


Received 6 April 2022; revised 24 July 2023; accepted 18 September 2023