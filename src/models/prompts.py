def brca_prompts():
    prompts = [
        # Labels
        [
            'invasive ductal carcinoma',
            'breast invasive ductal carcinoma',
            'invasive ductal carcinoma of the breast',
            'invasive carcinoma of the breast, ductal pattern',
            'idc'
            ],
        [
            'invasive lobular carcinoma',
            'breast invasive lobular carcinoma',
            'invasive lobular carcinoma of the breast',
            'invasive carcinoma of the breast, lobular pattern',
            'ilc'
            ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Breast Tissue Cells
        [
            'normal breast tissue',
            'normal breast cells',
            'normal breast',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def brca_her2_prompts():
    prompts = [
        # Labels
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Breast Tissue Cells
        [
            'normal breast tissue',
            'normal breast cells',
            'normal breast',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates

def brca_er_prompts():
    prompts = [
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Breast Tissue Cells
        [
            'normal breast tissue',
            'normal breast cells',
            'normal breast',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def brca_pr_prompts():
    prompts = [
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',

        ],
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Breast Tissue Cells
        [
            'normal breast tissue',
            'normal breast cells',
            'normal breast',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def nsclc_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
            ],
        [
            'squamous cell carcinoma',
            'lung squamous cell carcinoma',
            'squamous cell carcinoma of the lung',
            'lusc',
            ],

        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Lung Tissue Cells
        [
            'normal lung tissue',
            'normal lung cells',
            'normal lung',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)

    return cls_templates


def crc_msi_prompts():
    prompts = [
        # Labels
        [
            'colorectal carcinoma',
            'colorectal cancer',
            'crc',
            ],
        [
            'colorectal carcinoma',
            'colorectal cancer',
            'crc',
            ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Tissue Cells
        [
            'normal colon tissue',
            'normal colon cells',
            'normal colon',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)

    return cls_templates

def crc_braf_prompts():
    prompts = [
        # Labels
        [
            'colorectal carcinoma',
            'colorectal cancer',
            'crc',
            ],
        [
            'colorectal carcinoma',
            'colorectal cancer',
            'crc',
            ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Tissue Cells
        [
            'normal colon tissue',
            'normal colon cells',
            'normal colon',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)

    return cls_templates

def crc_tp53_prompts():
    prompts = [
        # Labels
        [
            'colorectal carcinoma',
            'colorectal cancer',
            'crc',
            ],
        [
            'colorectal carcinoma',
            'colorectal cancer',
            'crc',
            ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
            ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Tissue Cells
        [
            'normal colon tissue',
            'normal colon cells',
            'normal colon',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)

    return cls_templates


def luad_egfr_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
        ],
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
        ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Breast Tissue Cells
        [
            'normal lung tissue',
            'normal lung cells',
            'normal lung',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def luad_stk11_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
        ],
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
        ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
            ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
            ],
        # Normal Breast Tissue Cells
        [
            'normal lung tissue',
            'normal lung cells',
            'normal lung',
            ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def brca_pik3ca_prompts():
    prompts = [
        # Labels
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
        ],
        # Normal Breast Tissue Cells
        [
            'normal breast tissue',
            'normal breast cells',
            'normal breast',
        ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates

def brca_cdh1_prompts():
    prompts = [
        # Labels
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        [
            'breast cancer',
            'breast tumor',
            'breast carcinoma',
        ],
        # Adipocytes
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ],
        # Connective tissue
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        # Necrotic Tissu
        [
            'necrotic tissue',
            'necrosis',
        ],
        # Normal Breast Tissue Cells
        [
            'normal breast tissue',
            'normal breast cells',
            'normal breast',
        ],
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def stad_msi_prompts():
    prompts = [
        # Labels
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        [
            'normal gastric mucosa',
            'normal gastric tissue',
        ],
        [
            'necrotic tissue',
            'necrosis',
        ],
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ]
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)

    return cls_templates


def stad_ebv_prompts():
    prompts = [
        # Labels
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        [
            'normal gastric mucosa',
            'normal gastric tissue',
        ],
        [
            'necrotic tissue',
            'necrosis',
        ],
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ]
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)

    return cls_templates


def stad_lauren_prompts():
    prompts = [
        # Labels
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        [
            'normal gastric mucosa',
            'normal gastric tissue',
        ],
        [
            'necrotic tissue',
            'necrosis',
        ],
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ]
    ]

    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates

def stad_tp53_prompts():
    prompts = [
        # Labels
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        [
            'normal gastric mucosa',
            'normal gastric tissue',
        ],
        [
            'necrotic tissue',
            'necrosis',
        ],
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ]
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def stad_muc16_prompts():
    prompts = [
        # Labels
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'stomach adenocarcinoma',
            'stomach cancer',
            'stomach carcinoma',
        ],
        [
            'connective tissue',
            'stroma',
            'fibrous tissue',
            'collagen',
        ],
        [
            'normal gastric mucosa',
            'normal gastric tissue',
        ],
        [
            'necrotic tissue',
            'necrosis',
        ],
        [
            'adipocytes',
            'adipose tissue',
            'fat cells',
            'fat tissue',
            'fat',
        ]
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates

