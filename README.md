# Jarros de Água Generalizados

Aplicação Streamlit para resolver o problema dos jarros de água com vários jarros, capacidades configuráveis, objectivos flexíveis e visualização passo a passo.

## Executar

```powershell/cmd
pip install -r requirements.txt
streamlit run app.py
ou se estiver com problemas para importar o streamlit. pode instalar anaconda e adicione streamlitenv 
activate base
C:\o diretorio completo onde esta o projecto\ e depois clica shift+enter

e escreve streamlit run app.py 
C:\Users\kenym\OneDrive\Documents\ISCTEM\IA\TrabPrt>streamlit run app.py
```

## Funcionalidades

- Definir número de jarros, capacidades e estado inicial.
- Objectivo por quantidade num jarro, soma de jarros seleccionados, qualquer combinação ou estado exacto.
- Acções: encher, esvaziar e transferir água entre jarros.
- Algoritmos: BFS, DFS, custo uniforme e A*.
- Custo por número de operações ou por volume total movimentado.
- Sequência de operações, estados por passo, nós explorados, tempo e volume movimentado.
- Comparação automática entre solução com menor número de passos e solução com menor volume movimentado.
- Detecção automática de objectivos impossíveis quando o espaço alcançável pode ser provado.
- Árvore parcial de busca com destaque para estados da solução e estados objectivo.
