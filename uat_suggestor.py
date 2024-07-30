from uat_eval.uat_retriver import uat_retriever
from uat_eval.vertexai_llm import get_response
from uat_eval.similartity_search.similarity_search import search_uat
import ast
def uat_manager(abstract):
    # print(abstract)
    
    prompt = f'''
You are an expert astronomical research assistant tasked with extracting main concepts from abstracts of astronomical research papers. Your goal is to identify and list the key astronomical concepts, theories, and phenomena mentioned in these abstracts.
Follow these steps to extract the main concepts:
1. Carefully read the entire abstract.
2. Identify key astronomical concepts, which may include:
   - Celestial objects (e.g., black holes, neutron stars, exoplanets)
   - Astronomical phenomena (e.g., supernovae, gamma-ray bursts, gravitational waves)
   - Theoretical constructs (e.g., dark matter, dark energy, inflation)
   - Observational techniques or instruments (e.g., spectroscopy, interferometry, space telescopes)
   - Physical processes (e.g., accretion, nucleosynthesis, stellar evolution)
   - Astronomical metrics or properties (e.g., redshift, metallicity, orbital period)
3. For each identified concept:
   - Provide a brief (1-2 sentence) explanation of the concept in the context of the research.
   - If the concept is novel or uniquely applied in this research, note that in the explanation.
4. Output Format:
   Present your analysis in the following JSON structure:
   
   ```json
   {{"concepts": [
       {{
         "term": "Concept 1",
         "explanation": "Brief explanation of Concept 1 in the context of the research"
       }},
       {{
         "term": "Concept 2",
         "explanation": "Brief explanation of Concept 2 in the context of the research"
       }},
       ...
     ]
   }}
   ```
Remember to:
- Focus on astronomical concepts directly relevant to the research presented in the abstract.
- Provide clear, concise explanations that relate the concept to the specific research context.
- Include any novel applications or interpretations of known concepts if present in the abstract.
- Aim for scientific accuracy and objectivity in your explanations.
- If a concept is mentioned but not explained in the abstract, provide a general explanation based on common astronomical knowledge.
Ensure the "concepts" list in the JSON output is populated with all relevant concepts from the abstract, using an empty list ([]) only if no clear astronomical concepts are identifiable.

abstract:{abstract}'''
    output = get_response(custom_prompt=prompt)

    import json

    def clean_json_string(input_string):
        lines = input_string.split('\n')
        start = 0
        end = len(lines)

        # Remove opening ```json if present
        if lines[0].strip() == '```json':
            start = 1

        # Remove closing ``` if present
        if lines[-1].strip() == '```':
            end = -1

        return '\n'.join(lines[start:end])
    # Usage:
    json_string = clean_json_string(output)
    concepts_ = json.loads(json_string)

    list_of_terms = [concepts_['term'] for concepts_ in concepts_['concepts']]
    # list_of_terms = tokenizer.tokenize('Measuring the relation between star formation and galactic winds is observationally difficult. In this work we make an indirect measurement of the mass-loading factor (the ratio between the mass outflow rate and star formation rate) in low-mass galaxies using a differential approach to modeling the low-redshift evolution of the star-forming main sequence and mass–metallicity relation. We use Satellites Around Galactic Analogs (SAGA) background galaxies, i.e., spectra observed by the SAGA Survey that are not associated with the main SAGA host galaxies, to construct a sample of 11,925 spectroscopically confirmed low-mass galaxies from 0.01 ≲ z ≤ 0.21 and measure auroral line metallicities for 120 galaxies. The crux of the method is to use the lowest-redshift galaxies as the boundary condition of our model, and to infer a mass-loading factor for the sample by comparing the expected evolution of the low-redshift reference sample in stellar mass, gas-phase metallicity, and star formation rate against the observed properties of the sample at higher redshift. We infer a mass-loading factor of ${\eta }_{{\rm{m}}}={0.92}_{-0.74}^{+1.76}$ , which is in line with direct measurements of the mass-loading factor from the literature despite the drastically different sets of assumptions needed for each approach. While our estimate of the mass-loading factor is in good agreement with recent galaxy simulations that focus on resolving the dynamics of the interstellar medium, it is smaller by over an order of magnitude than the mass-loading factor produced by many contemporary cosmological simulations.')
    terms_string = ' '.join(list_of_terms)
    branches_, keywords_,scores_uat_ = uat_retriever(terms_string,top_k=10)
    branches, keywords,scores_uat = uat_retriever(abstract,top_k=10)
    abstracts,dis,titles,scores_ads = search_uat(abstract)
    
    if max(scores_uat.values()) > 0.55:
        # print(max(scores_uat.values()))
        # print(f"Paper has a concept similar to the UAT, nearest branches are")
        context = f'''Given abstract:
{abstract}
Given set of UAT branches from abstract:
1. {branches[0]}
2. {branches[1]}
3. {branches[2]}
4. {branches[3]}
5. {branches[4]}
6. {branches[5]}
7. {branches[6]}
8. {branches[7]}
9. {branches[8]}
10. {branches[9]}
Given set of UAT branches from key concepts of abstract:
1. {branches_[0]}
2. {branches_[1]}
3. {branches_[2]}
4. {branches_[3]}
5. {branches_[4]}
6. {branches_[5]}
7. {branches_[6]}
8. {branches_[7]}
9. {branches_[8]}
10. {branches_[9]}'''
        responce = get_response(prompt_name = "reranker",input_data=context)
        # print(responce)
        responce_dict = ast.literal_eval(responce.split("```json\n")[1].split("\n```")[0])
        list_of_names = [responce_dict['top_branches'][i]['name'] for i in range(10)]
        # print(responce_dict)
        output = f'''UAT Branches in order of relevance:
1.  Branch : {responce_dict['top_branches'][0]['name']}
    Relevance : {responce_dict['top_branches'][0]['relevance']}\n
2.  Branch : {responce_dict['top_branches'][1]['name']}
    Relevance : {responce_dict['top_branches'][1]['relevance']}\n
3.  Branch : {responce_dict['top_branches'][2]['name']}
    Relevance : {responce_dict['top_branches'][2]['relevance']}\n
4.  Branch : {responce_dict['top_branches'][3]['name']}
    Relevance : {responce_dict['top_branches'][3]['relevance']}\n
5.  Branch : {responce_dict['top_branches'][4]['name']} 
    Relevance : {responce_dict['top_branches'][4]['relevance']}\n
6.  Branch : {responce_dict['top_branches'][5]['name']}
    Relevance : {responce_dict['top_branches'][5]['relevance']}\n
7.  Branch : {responce_dict['top_branches'][6]['name']}
    Relevance : {responce_dict['top_branches'][6]['relevance']}\n
8.  Branch : {responce_dict['top_branches'][7]['name']}
    Relevance : {responce_dict['top_branches'][7]['relevance']}\n
9.  Branch : {responce_dict['top_branches'][8]['name']}
    Relevance : {responce_dict['top_branches'][8]['relevance']}\n
10. Branch : {responce_dict['top_branches'][9]['name']}
    Relevance : {responce_dict['top_branches'][9]['relevance']}'''
        # print(output)
        output_dict = {"type":"reranked",
                        "output":output,
                        "list_of_branch":list_of_names,
                        "retreived_branches_abstract":branches,
                        "retreived_branches_concepts":branches_,
                        "retreived_keywords_abstract":concepts_}
        return output_dict


    elif max(scores_ads) > 0.80:
        
        context = f'''Primary Abstract:
{abstract}
Similar Abstracts:
1. {abstracts[0]}
2. {abstracts[1]}
3. {abstracts[2]}
4. {abstracts[3]}
5. {abstracts[4]}
Top 5 similar UAT concept branches from abstract:
1. {branches[0]}
2. {branches[1]}
3. {branches[2]}
4. {branches[3]}
5. {branches[4]}
Top 5 similar UAT branches from key concepts of abstract:
1. {branches_[0]}
2. {branches_[1]}
3. {branches_[2]}
4. {branches_[3]}
5. {branches_[4]}'''
        result = get_response(prompt_name = "concept_suggestor 3",input_data=context)
        result_dict = ast.literal_eval(result.split("```json\n")[1].split("\n```")[0])
        with open("new_concept.txt","w") as f:
            f.write(result)
        new_concept = result_dict['most_relevant_new_addition']['name']
        new_concept_definition = result_dict['most_relevant_new_addition']['definition']
        branches_concept, keywords,scores_uat = uat_retriever(new_concept,top_k=10)
        if max(scores_uat.values()) >= 0.6:
            output_dict = {"type":"new_concept_exists",
                           "output":f"New concept {new_concept} is already present in the UAT,nearest branches are {branches_concept[0]}, {branches_concept[1]}, {branches_concept[2]}, {branches_concept[3]}, {branches_concept[4]}",
                            "list_of_branch":branches_concept,
                            "retreived_branches_abstract":branches,
                            "retreived_branches_concepts":branches_,
                            "retreived_keywords_abstract":concepts_}
            return output_dict
        elif max(scores_uat.values()) < 0.6:
            context = f'''New Concept: {new_concept}
Definition: {new_concept_definition}

Relevant UAT Branches from abstract:
1. {branches[0]}
2. {branches[1]}
3. {branches[2]}
4. {branches[3]}
5. {branches[4]}
6. {branches[5]}
7. {branches[6]}
8. {branches[7]}
9. {branches[8]}
10. {branches[9]}
Relevant UAT branches from key concepts of abstract:
1. {branches_[0]}
2. {branches_[1]}
3. {branches_[2]}
4. {branches_[3]}
5. {branches_[4]}
6. {branches_[5]}
7. {branches_[6]}
8. {branches_[7]}
9. {branches_[8]}
10. {branches_[9]}'''
            concept_result = get_response(prompt_name = "branch_suggestor",input_data=context)
            concept_dict = ast.literal_eval(concept_result.split("```json\n")[1].split("\n```")[0])
            name_concept = concept_dict['new_concept']['name']
            output = f'''New Concept : {concept_dict['new_concept']['name']}\n
Definition : {concept_dict['new_concept']['definition']}\n
Suggested Placement :\n
Branch : {concept_dict['suggested_placement']['branch']}\n
Location : {concept_dict['suggested_placement']['location']}\n
Justification : {concept_dict['suggested_placement']['justification']}'''
            output_dict = {"type":"new_concept",
                           "output":output,
                        #    "name_concept":name_concept,
                           "list_of_branch":concept_dict,
                            "retreived_branches_abstract":branches,
                            "retreived_branches_concepts":branches_,
                            "retreived_keywords_abstract":concepts_}
            return output_dict
    else:
        output_dict = {"type":"no_match",
                       "output":"Paper has a different concept, please add it to the UAT manually.",
                       "list_of_branch":None,
                       "retreived_branches_abstract":branches,
                       "retreived_branches_concepts":branches_,
                       "retreived_keywords_abstract":concepts_}
        return output_dict




# uat_manager('Gravitational wave (GW) detections of binary neutron star inspirals will be crucial for constraining the dense matter equation of state (EOS). We demonstrate a new degeneracy in the mapping from tidal deformability data to the EOS, which occurs for models with strong phase transitions. We find that there exists a new family of EOS with phase transitions that set in at different densities and that predict neutron star radii that differ by up to ∼500 m but that produce nearly identical tidal deformabilities for all neutron star masses. Next-generation GW detectors and advances in nuclear theory may be needed to resolve this degeneracy.')
# print("\n")
# uat_manager('In this paper, we introduce a novel data augmentation methodology based on Conditional Progressive Generative Adversarial Networks (CPGAN) to generate diverse black hole (BH) images, accounting for variations in spin and electron temperature prescriptions. These generated images are valuable resources for training deep learning algorithms to accurately estimate black hole parameters from observational data. Our model can generate BH images for any spin value within the range of [-1, 1], given an electron temperature distribution. To validate the effectiveness of our approach, we employ a convolutional neural network to predict the BH spin using both the GRMHD images and the images generated by our proposed model. Our results demonstrate a significant performance improvement when training is conducted with the augmented data set while testing is performed using GRMHD simulated data, as indicated by the high R2 score. Consequently, we propose that GANs can be employed as cost-effective models for black hole image generation and reliably augment training data sets for other parametrization algorithms.')