New capabilities will be added to the api.
Api code will be refactored to reflect this new scope.


New endpoints:

##Endpoint: /compliance-map 
For each section of the policy, determine which statutes apply to the section of the policy and whether the policy section is compliant or non-compliant, or neither.
*Parameters:
database
source_collection
source_document_id
target_collection
target_document_id

target is customer policy. source is the statute. 


##Endpoint: /peer-compliance
*Parameters:
database
source_collection
source_document_id
target_collection
target_document_id

source is the peer policies.  target is the customer policy.


##Endpoint: /industry-compliance
*Parameters:
database
source_collection
source_document_id
target_collection
target_document_id

source is the Nymity Privacy Management Accountability Framework or ISO/IEC 27701.  target is the customer policy.


Walk through how I would perform a comparison of a privacy policy to the statute to audit the compliance of the statute.
I have the statutes and policy in a vectored database.
Some of things I want:
- For each section of the policy, determine which statutes apply to the section of the policy and whether the policy section is compliant or non-compliant, or neither.

Suggest other comparisons that are standard in the privacy audit industry.



Compare your current policy against comprehensive frameworks like the Nymity Privacy Management Accountability Framework or ISO/IEC 27701 to identify missing controls.

 Compare your disclosures against industry peers to ensure your transparency level meets market standards and consumer expectations.




Use c