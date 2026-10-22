import re

#目标图构建器
class GraphBuilder:
    """从文本描述构建目标图"""
    # 非物体停用词——LLM 可能从描述文本中错误提取出的通用词
    STOPWORDS = {
        'image', 'picture', 'photo', 'center', 'object', 'item', 'thing',
        'none', 'null', 'unknown', 'background', 'scene', 'room', 'area',
        'space', 'view', 'angle', 'side', 'part', 'the', 'a', 'an',
    }
    def __init__(self, llm=None):
        self.llm = llm

    def _is_valid_object(self, name):
        """检查提取的名称是否为有效物体（非停用词、非空、长度>=2）"""
        name = name.lower().strip().strip('"').strip("'").strip('.')
        if not name or len(name) < 2:
            return False
        return name not in self.STOPWORDS

    def get_objects(self, llm_response):
        objects = llm_response.split('[')[1].split(']')[0].split(',')
        objects = objects[:5]
        cleaned = []
        for obj in objects:
            obj = obj.strip().strip('"').strip("'").strip('.')
            if self._is_valid_object(obj):
                cleaned.append(obj)
        return cleaned

    def _extract_objects_fallback(self, llm_response):
        """Regex fallback: extract quoted or comma-separated object names
        when the LLM response lacks the expected [obj1, obj2, ...] format."""
        # Try quoted strings first: "chair", 'table', etc.
        quoted = re.findall(r'["\']([^"\']{2,30})["\']', llm_response)
        if quoted:
            return [q.strip() for q in quoted if self._is_valid_object(q)]
        # Try numbered list: 1. chair  2. table
        numbered = re.findall(r'\d+[.)\s]+([A-Za-z][A-Za-z ]{1,30})', llm_response)
        if numbered:
            return [n.strip() for n in numbered if self._is_valid_object(n)]
        # Try comma-separated bare words from the whole response
        tokens = [t.strip().strip('.\'"') for t in llm_response.split(',')]
        valid = [t for t in tokens if self._is_valid_object(t) and len(t.split()) <= 3]
        return valid[:5]

    def get_relations(self, llm_response, objects):
        relations = []
        for line in llm_response.split('\n'):
            relation = {'source': '', 'target': '', 'type': ''}
            parts = line.strip().split(': ')
            if len(parts) == 2:
                relation_info = parts[1].strip()
                relation_parts = relation_info.split(' is ')
                if len(relation_parts) == 2:
                    source_target = parts[0].strip().split(' and ')
                    if len(source_target) == 2:
                        source = source_target[0].strip()
                        target = source_target[1].strip()
                        relation_type = relation_parts[1].strip()
                        if source in objects and target in objects:
                            relation = {'source': source, 'target': target, 'type': relation_type}
                            relations.append(relation)
        return relations

    def parse_text_description(self, description, max_retries=3):
        """解析文本描述，提取对象和关系（带重试 + regex fallback）"""
        #步骤1：LLM提取对象（最多重试 max_retries 次）
        object_prompt = f"""
        Please extract all concrete physical objects (furniture, appliances, fixtures, etc.) mentioned in the following text description.
        Do NOT include generic words like "image", "object", "center", "item", "scene", or "room".
        Output format: "[<object 1>, <object 2>,...]"
        Text Description: {description}
        """
        objects = []
        for attempt in range(max_retries):
            object_response = self.llm(object_prompt)
            try:
                objects = self.get_objects(object_response)
            except Exception:
                # Bracket-based parsing failed; try regex fallback
                objects = self._extract_objects_fallback(object_response)
            if objects:
                break
            print(f"  [GraphBuilder] Object extraction attempt {attempt+1}/{max_retries} failed, response: {object_response[:120]}")

        if not objects:
            print(f"  [GraphBuilder] WARNING: All extraction attempts failed for: {description[:80]}")

        #步骤2：LLM提取关系
        relations = []
        if len(objects) >= 2:
            relation_prompt = f"""
            Please describe the relationships between the above objects in the following text description and provide the information in the specified format.
            The format should be: "<Object A> and <Object B>: <Object A> is <relation type> <Object B>".
            If there are multiple relationships, please list them one per line.

            Text Description: {description}

            Example output format:
            Book and Table: Book is on Table
            """
            relation_response = self.llm(relation_prompt)
            relations = self.get_relations(relation_response, objects)

        return objects, relations

    def build_graph(self, objects, relations):
        """构建图数据结构"""
        graph = {
            'nodes': [{'id': obj} for obj in objects],
            'edges': [{'source': r['source'], 'target': r['target'], 'type': r['type']} for r in relations]
        }
        return graph

    def build_graph_from_text(self, text_goal):
        objects, relations = self.parse_text_description(text_goal)
        graph = self.build_graph(objects, relations)
        return graph
