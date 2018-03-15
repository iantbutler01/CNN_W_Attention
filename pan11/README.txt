
Release Notes
PAN-11 AUTHORSHIP TRACK CORPUS

This is the preliminary release corpus for the PAN-11 authorship
attribution track.  This track comprises 7 tests for 5 different
training sets.  The training sets are:

Name	 Number of Authors   Number of Documents
--------------------------------------------------
Large	    72	   	        9337
Small	    26		        3001
Verify1	     1		          42
Verify2	     1		       	  55
Verify3	     1		       	  47

For each of the Large and Small training sets, there are two tests,
one only containing authors in the training set, and one containing
also around 20 other out-of-training authors.  All of the verification
test sets inherently include out-of-training authors. In this
preliminary release, you are given example test sets, called "Valid"
(for validation) sets; in each case, if the test set contains
out-of-training authors, the name ends in a +.  For the files provided
in this package, the statistics are:

Name	       Number of Authors   Number of Documents
-------------------------------------------------------
LargeValid	  66			1298
LargeValid+	  86			1440
SmallValid	  23			 518
SmallValid+	  43			 601
Verify1Valid+	  24			 104
Verify2Valid+	  21			 95
Verify2Valid+	  23			 100

For each of these a testing file, not containing author IDs, is given
as well as a ground truth file, containing the actual author IDs for
the texts, for validation purposes.

Note that apart from name redaction (as mentioned below), the texts are
intended to reflect a natural task environment, and so there are some texts,
both in training and in testing sets, that are not in English, or that are
automatically generated. You need not give an answer for all test documents,
which may reduce recall, but may increase your precision.

Each of the files is in an XML format, with similar schemas, as
follows.  The training files look like:

<training>
 <text file="<some unique filename>">
  <author id="<unique author ID>"/>
  <body>
TEXT OF THE MESSAGE
  </body>
 </text>
 ...
</training>

Testing files look like:

<testing>
 <text file="<some unique filename>">
  <body>
TEXT OF THE MESSAGE
  </body>
 </text>
 ...
</testing>

And the ground truth files look like:

<results>
 <text file="<some unique filename>">
  <author id="<unique author ID>"/>
 </text>
 ...
</results>

Submitted results must be in the ground truth file format for
evaluation.

Most personal names and email addresses have been (automatically)
redacted, and replaced (on a token-by-token basis) by <NAME/> and
<EMAIL/> tags, respectively.  This redaction is admittedly imperfect,
but we do not recommend relying on its imperfections.  Other than this
redaction, each text is typographically identical to the original
electronic text, so you can, in principle, rely on line length,
punctuation, and the like.

Please contact Shlomo Argamon via our internal mailing list pan@webis.de
or via our public mailing list pan-workshop-series@googlegroups.com if you
have any questions.
