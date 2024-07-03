import os
from dotenv import load_dotenv
import re
import json
from openai import OpenAI

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def conclude(text):
	response = client.chat.completions.create(
	model="gpt-4-turbo",
	messages=[
		{
		"role": "system",
		"content": "You are a knowledgeable assistant with expertise in medical imaging and diagnostics. You are assisting a user in analyzing Brain MRI scans. Your task is to generate a concise conclusion based on the scan data. You should always ensure that the conclusion is clear and medically accurate. You will be given observation in the following patter: [Observation] <Observation data goes here>. You need to return only the conclusion in plain text format."
		},
		{
		"role": "user",
		"content": "[Observation]\nScanning technique: T1 FSE-sagital, T2 FLAIR, T2 FSE-axial, T2 FSE-coronar, DWI.\nOn a series of tomograms in the structure of the subcortical and periventricular white matter of the cerebral hemispheres on both sides, multiple round-shaped foci of pathological intensity are asymmetrically determined, with unclear contours, homogeneous hyperintense signal characteristics on T2 VI and T2 FLAIR, with a diameter of 3-10 mm.\nThe basal ganglia are usually located, symmetrical, with clear, even contours, the dimensions are not changed. The MR signal of the nuclei was not changed.\nThe longitudinal fissure of the cerebrum is located centrally. Convexital grooves are moderately widened in the frontotemporal regions on both sides due to atrophy of the adjacent parts of the brain\nThe lateral ventricles are symmetrical, the width of the ventricles at the level of the foramen of Monro is 10 mm on the right, 14 mm on the left. The third ventricle is 4 mm wide. Sylvian aqueduct has not been changed. The fourth ventricle is tent-shaped and not dilated.\nThe corpus callosum is of normal shape and size. The brain stem is without areas of pathological intensity. The cerebellum is of normal shape, the signal characteristics of the white matter are not changed. The width of the cerebellar cortex is reduced due to atrophy. The craniovertebral junction is unchanged.\nThe pituitary gland is moderately flattened, the height in the sagittal projection is 3 mm. The pituitary stalk is located centrally. The chiasma of the optic nerves is located usually, the contours are clear and even. Parasellar cisterns without areas of pathological intensity. The siphons of the internal carotid artery are not changed. The cavernous sinuses of both carotid arteries are symmetrical, with clear, even contours.\nThe shape of the orbital cones on both sides is unchanged. The eyeballs are spherical in shape and of normal size. The signal characteristics of the vitreous body are not changed. The diameter of the optic nerves is not changed. The perineural subarachnoid space of the orbits is moderately diffusely dilated. The extraocular muscles are of normal size, the structure is without pathological signals. Retrobulbar fatty tissue without pathological signals.\nRegion of the cerebellopontine angle: the prevestocochlear nerve is clearly differentiated on both sides. Pneumatization of the mastoid processes of the temporal bones is preserved.\nA cyst with a diameter of 14 mm is identified in the left maxillary sinus. The remaining paranasal sinuses are usually pneumatized."
		},
		{
		"role": "assistant",
		"content": "MRI signs of vascular encephalopathy with the presence of multiple ischemic foci and atrophy of the frontotemporal areas on both sides. Left maxillary sinus cyst.\n\nConsultation with a neurologist is recommended.\nConsultation with an otolaryngologist is recommended."
		},
		{
		"role": "user",
		"content": f"""[Observation]

						{text}

					"""
		}
	],

	temperature=1,
	max_tokens=128,
	top_p=1,
	frequency_penalty=0,
	presence_penalty=0
	)

	response_text = response.choices[0].message.content
	# print('\n=====================\n')
	# print("User request_text: ", text)
	# print('\n')
	# print("OpenAI response_text (conclusion): ", response_text)
	# print('\n=====================\n')

	return response_text

# Usage 
# text = """
# Scanning technique: T1 TSE dark fluid-sagital, T2 TSE dark fluid, T2 TSE-axial, T2 TSE-coronar, DIW, 3D TOF 3 sl.\nOn a series of tomograms, the longitudinal fissure of the cerebrum is located centrally. Convexital grooves are not expanded, their number is not changed. The thickness of the cerebral cortex is preserved, the distribution of gray matter is not disturbed.\nThe lateral ventricles are symmetrical, the width of the ventricles at the level of the foramen of Monro is 8 mm on the right, 8 mm on the left. The third ventricle is 4 mm wide. Sylvian aqueduct has not been changed. The fourth ventricle is tent-shaped and not dilated. The cisterna magna is dilated with retrocerbellar extension.\nThe white matter of the brain has homogeneous isointense signal characteristics; the signal intensity in the periventricular areas is not changed. The basal ganglia are usually located, symmetrical, with clear, even contours, the dimensions are not changed, the MR signal is not changed. The corpus callosum is of normal shape and size. The brain stem is without features. The cerebellum is of normal shape, the signal characteristics of the white matter are not changed. The width of the cerebellar cortex is preserved. The craniovertebral junction is unchanged.\nThe pituitary gland is of normal shape, height in the sagittal projection is 3.5 mm. The pituitary stalk is located centrally. The chiasma of the optic nerves is located usually, the contours are clear and even. Parasellar cisterns without areas of pathological intensity. The siphons of the internal carotid arteries are not changed. The cavernous sinuses of both carotid arteries are symmetrical, with clear, even contours.\nThe shape of the orbital cones on both sides is unchanged. The eyeballs are spherical in shape and of normal size; the MR signal of the vitreous body is not changed. The diameter of the optic nerves was preserved. The perineural subarachnoid space of the orbits is moderately diffusely dilated. The extraocular muscles are of normal size, the structure is without pathological signals. Retrobulbar fatty tissue without pathological signals.\nRegion of the cerebellopontine angle: the prevestocochlear nerve is clearly differentiated on both sides. Pneumatization of the mastoid processes of both temporal bones is not impaired. The paranasal sinuses are usually pneumatized. The nasal turbinates are hypertrophied. The nasal septum is deviated to the left.\nOn angiograms of intracranial vessels performed in 3D TOF 3 sl mode, the internal carotid arteries are symmetrical, can be traced along their entire length, their course and caliber are not changed. The siphons of the carotid arteries on both sides are not displaced, without signs of external compression. The intraluminal signal is intense and homogeneous. The MCA on both sides arise normally, with normal segmental division and intraluminal signal, without lumen narrowing or displacement. The stroke and caliber of the PMA has been preserved. The basal artery has a normal course and caliber. The PMA usually recedes, the stroke and caliber are not changed. Anterior and posterior communicating arteries, on both sides without features.\nIn the field of view: In the structure of the left parotid gland, a cystic area with clear, even contours is determined, with signal characteristics that are hyperintense on T2-WI and FLAIR, hypointense on T1-WI.
# """

# print(conclude(text))