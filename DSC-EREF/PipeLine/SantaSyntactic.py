import stanza
import json
stanza.download('zh')
nlp = stanza.Pipeline(lang='zh', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False)

def extract_spo(sentence: str):
    doc = nlp(sentence)
    results = []

    for sent in doc.sentences:
        words = sent.words
        word_text = [w.text for w in words]
        roots = [w for w in words if w.deprel == 'root']
        if not roots:
            roots = [w for w in words if w.upos == 'VERB']
        for root in roots:
            spo = {}
            spo['谓语'] = root.text
            subjects = [w.text for w in words if w.head == root.id and w.deprel in ('nsubj', 'csubj')]
            if subjects:
                spo['主语'] = ''.join(subjects)
            objects = [w.text for w in words if w.head == root.id and w.deprel in ('obj',)]
            if objects:
                spo['宾语'] = ''.join(objects)
            adverbials = []
            for w in words:
                if w.head == root.id and w.deprel in ('advmod', 'obl'):
                    phrase = _expand_phrase(w, words)
                    adverbials.append(phrase)
                elif w.deprel == 'case' and any(ww.head == w.head and ww.id == root.id for ww in words):
                    head_noun = next((ww for ww in words if ww.id == w.head), None)
                    if head_noun:
                        phrase = _expand_phrase(head_noun, words)
                        adverbials.append(w.text + phrase)
            if adverbials:
                spo['状语'] = ''.join(adverbials)
            complements = []
            for obj in [w for w in words if w.head == root.id and w.deprel == 'obj']:
                comps = [c.text for c in words if c.head == obj.id and c.deprel in ('amod', 'nmod', 'acl')]
                if comps:
                    complements.append(''.join(comps))
            for subj in [w for w in words if w.head == root.id and w.deprel in ('nsubj', 'csubj')]:
                comps = [c.text for c in words if c.head == subj.id and c.deprel in ('amod', 'nmod', 'acl')]
                if comps:
                    complements.append(''.join(comps))
            if complements:
                spo['补语'] = ''.join(complements)
            parts = []
            for role in ['主语', '状语', '谓语', '宾语', '补语']:
                if role in spo:
                    parts.append(f"{role}:{spo[role]}")
            if parts:
                results.append("[" + ",".join(f"'{p}'" for p in parts) + "]")

    return results

def _expand_phrase(center_word, words):
    phrase = [center_word.text]
    for w in words:
        if w.head == center_word.id and w.deprel in ('det', 'nummod', 'amod', 'nmod'):
            phrase.insert(0, w.text)
    for w in words:
        if w.head == center_word.id and w.deprel in ('clf', 'acl', 'nmod'):
            phrase.append(w.text)
    return ''.join(phrase)

data = json.load(open("../Data/ori.json", 'r', encoding="utf-8"))#字典列表
results = []
for i, dic in enumerate(data):
    ori_text = dic['ori_text']  # 原文本
    spo_list = []
    spo_list.extend(extract_spo(ori_text))
    new_dic = {}
    new_dic['idx'] = dic['idx']
    new_dic['ori_text'] = ori_text
    new_dic['spo_list'] = spo_list
    print(f"{i}th already!")
    results.append(new_dic)
    print(results)
with open(f"../Result/santasyntactic.json", "w", encoding='utf-8') as f:
    f.write(json.dumps(results, indent=2, ensure_ascii=False))
f.close()