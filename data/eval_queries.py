"""
eval_queries.py
-----------------
A small hand-labelled evaluation set for the retrieval POC.

Each entry is (query, expected_doc_id). Relevance is judged at the
DOCUMENT level (i.e. "did the retriever return a chunk from the correct
source document"), not the exact chunk, because with 150-word overlapping
chunks a query's answer can legitimately span more than one chunk of the
same document. This is a coarser but more robust ground truth for a
22-query POC than manually pinning exact chunk ids.

doc_id values must match the filenames (without extension) under
data/raw_corpus/.
"""

EVAL_SET = [
    # registration_booklet_2020
    ("How much is the Student Contribution Fee for 2020/21?", "registration_booklet_2020"),
    ("What happens if I want to defer my place for medical reasons?", "registration_booklet_2020"),
    ("What is the fee for the BA Hons in Psychology course?", "registration_booklet_2020"),
    ("When does Semester 1 start and end?", "registration_booklet_2020"),
    ("What support does the Mathematics Support Service provide?", "registration_booklet_2020"),
    ("How do I access online registration on the NCI website?", "registration_booklet_2020"),
    ("What transport options are available for getting to NCI?", "registration_booklet_2020"),

    # qa_handbook_ch6_admissions
    ("What is the Disability Access Route to Education (DARE)?", "qa_handbook_ch6_admissions"),
    ("How does the college handle appeals of admissions decisions?", "qa_handbook_ch6_admissions"),
    ("What is Recognition of Prior Learning (RPL)?", "qa_handbook_ch6_admissions"),
    ("What are the entry requirements for non-standard applications?", "qa_handbook_ch6_admissions"),
    ("How are interviews used in the admissions screening process?", "qa_handbook_ch6_admissions"),

    # library_referencing_guide
    ("How do I reference a journal article in Harvard style?", "library_referencing_guide"),
    ("What is the CRAAP test used for when evaluating sources?", "library_referencing_guide"),
    ("What counts as plagiarism and how can I avoid it?", "library_referencing_guide"),
    ("How should I use generative AI ethically in my assignments?", "library_referencing_guide"),
    ("What is the difference between paraphrasing and quoting?", "library_referencing_guide"),

    # results_faq_sob
    ("What do I do if I failed a module in Semester 1?", "results_faq_sob"),
    ("Is there a fee for the repeat contingency assessment after a Covid-19 deferral?", "results_faq_sob"),
    ("What is the deadline to register for the repeat contingency assessment?", "results_faq_sob"),

    # student_card_app_guide
    ("What are the photo requirements for my student ID card?", "student_card_app_guide"),
    ("How do I submit my student card photo from a desktop computer?", "student_card_app_guide"),
]
