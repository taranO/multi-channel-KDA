
## Machine learning through cryptographic glasses: combating adversarial attacks by key based diversified aggregation

The research was supported by the [SNF](http://www.snf.ch) project No. 200021_182063. 
##

The Key based Diversified Aggregation (KDA) mechanism as a defense strategy in a gray and black-box scenario. KDA assumes
that the attacker (i) knows the architecture of classifier and the used defense strategy, (ii) has an access to the training data set but (iii) does not know a secret key and does not have access to the internal states of the system. The
robustness of the system is achieved by a specially designed key based randomization. The proposed randomization prevents the gradients’ back propagation and restricts the attacker to create a ”bypass” system. The randomization is performed simultaneously in several channels. Each channel introduces its own randomization in a special transform domain. The sharing of a secret key between the training and test stages creates an information advantage to the defender. Finally, the aggregation of soft outputs from each channel stabilizes the results and increases the reliability of the final score. 

<p align="center">
<img src="http://sip.unige.ch/files/2115/8330/8350/multi-channel_KDA.png" height="450px" align="center">
<br/>
<br/>
Fig.1: Generalized diagram of the multi-channel system with the KDA. 
</p>
