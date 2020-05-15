# Hierarchical LSTM

The model was initially implemented to solve multiple interventions
problems for MOOC discussion forums.

## MOOC

MOOC stands for Massive Open Online Courses. It is a massive open online course
is an online course aimed at unlimited participation and open access via the web.

Source: Wikipedia

Some examples of MOOCs include Coursera and edX.

## Multiple Intervention Problems

Given a discussion forum thread containing a list of posts ordered from the earliest
timestamp to the latest timestamp (Note: This is one way to look at the problem. See
[postia](https://github.com/CT15/postia) for further details), determine which posts
in this thread require/merit instructor interventions. It is a binary classification
problem.

Instructor intervention is a term used to refer to instructor's reply to a particular
post.

## Model Architecture

![Architecture](/images/architecture.jpg "Hierarchical LSTM architecture")
