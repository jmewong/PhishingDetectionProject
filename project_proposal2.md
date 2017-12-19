# Identifying phishing websites

Based on a feature set that includes information of the url, content, and other miscellaneous behavior of the site, can we correctly identify whether or not it is a phishing website?

We will use statistics from https://archive.ics.uci.edu/ml/datasets/phishing+websites. We are considering some of the following features:

 * URL composition
 * Popup/redirection 
 * HTTPS
 * Domain registration length
 * Sub domains and multi sub domains

The question is significant due to the recent proliferation of phishing/scamming/virus/malware in the modern era. Everyone who has interacted with modern technology has come across and likely have been in danger of identity theft, losing sensitive personal information or money from this type of online scamming.

This project is worthwhile because we can create a model that will be a tool for discerning between fake and real websites for web users of all experience levels. Additionally, our data repository has an extensive feature list and a large amount of data points, and this should ensure enough information needed for our regression/classification methods.