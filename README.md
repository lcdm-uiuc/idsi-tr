# iDSI Core

Herein lies an accurate description of what the iDSI Core does and what is hosted in this repo.

## Technical Reports?

Yes, technical reports are a (convex) combination of research paper and a white paper. In a white paper, you typically detail what you do to get to the research paper, and research papers talk about findings and whatnot. The goal of the IDSI group is to produce a series of technical reports in order to educate users on how to set up different cloud computing architectures and their benefits. This could be anywhere from Professors setting up their research, industry professionals, people looking to learn more about the different cloud computing aspects.

# Writing the reports

## Topics

We have loads of topics to choose from right now we have the following ideas listed and the following people working on them.

* Apache Ambari
   * Quinn - Setting up Ambari
* Cassandra
   * Bhuvan - Introduction to Cassandra
* Fabric
   * Ben - Intro to Fabric
* Flink
* Hadoop
   * Nishil - Custom Data Types
   * Nishil - Processing Custom Inputs
   * Nishil - Output Formats in Hadoop
* Hive
* Kafka
* Kubernetes
* Mesos
* MongoDB
* Neo4J
* Pig
* Spark
   * Sameet and Josh - Hadoop, Spark, PySpark installation and configuration
* Spark GraphX
* Spark MLib

#### Datasets:

Financial data (200+ GB, Professor Brunner)
Financial Data (potentially from corporate partner, Multi-TB?, awaiting).

## Checklist

Make sure to 
* List prerequisite knowledge (Make sure that your technical report prerequisite knowledge doesn't create a cycle)
* Links to other technical reports
* List intended audience at beginning of report
* Links to additional information
* Versioning **TBD How to do**
* Shouldn’t read like a novel or be a written code repository

## Authorship

As a general note for authorship: if you use anything relating to the cluster/ambari put Quinn Jarrell in your report (consider linking as a pre-req). If you have any fabric scripts, link Ben Congdon in your report (consider linking that report as a pre req as well). All papers should have the original author and Prof. Robert Brunner as well as any organizations of which anybody is a part.

## Citations

When in doubt cite! [Cite: Ben Congdon]. We will use **TBD CITATION STYLE**. Make sure to have links to additional exploration.

## Writing

Okay, you've got your topic -- potentially dataset -- you've done some research and you are ready to start writing. What do you do? 

* First create a branch off the master branch. 
* Call it [technical_report]_tr. 
* Make a folder in the reports/ directory with what your technical report is. 
* Copy the template.tex in the root to the folder you just created.
* Write to your hearts content.
* Go to the root directory and type make, and you should see the technical report in your directory.

#### Well what do I write?

First, take a look at the rest of the stuff in the repository (try to pick something close to what you are writing if you are considering an intro-level report versus a report analyzing different aspects). Make sure to include a bit about setting up the data and how one should do that in general. Make sure to include benefits and drawbacks of this approach. Always consider your target audience when writing.

## Reviewing

Once you feel that you have a good first draft, go to the 

## Licensing of the Technical Reports and Code

**Prolly NCSA license, this part isn't entirely decided yet.**


#### Technical Report Style
Format borrowed from [Overleaf](https://www.overleaf.com/latex/templates/latex-template-for-preparing-an-article-for-submission-to-optica/gmsbdqxbmntw#.WIgCA7YrK35) which is licensed under the [LaTeX Project Public License](https://www.latex-project.org/lppl/)
