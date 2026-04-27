# Escoamento Monofásico Compressível em Meio Poroso (1D)
**Método dos Elementos Finitos com Firedrake**

---

## Descrição

Este projeto resolve numericamente o escoamento monofásico compressível do gás hidrogênio em um **meio poroso unidimensional**, utilizando o **Método dos Elementos Finitos (MEF)** com a biblioteca **Firedrake**.

O modelo considera propriedades termodinâmicas **variáveis com pressão e temperatura**, obtidas a partir de bibliotecas externas e interpoladas via tabelas pré-computadas.

---

## Objetivos do Projeto

- Resolver a equação de pressão para escoamento compressível em meio poroso;
- Incorporar **viscosidade μ(P,T)**, **densidade rho(P,T)** e **fator de compressibilidade Z(P,T)**;
- Construir e validar tabelas de propriedades termodinâmicas;
- Avaliar o impacto de propriedades reais no comportamento do escoamento.

---

## Hipóteses

- Meio poroso rígido (porosidade constante);
- Escoamento monofásico (gás hidrogênio);
- Ausência de termo fonte;
- Efeitos gravitacionais desprezados;
- Temperatura constante (nesta etapa do projeto).

---

## Estrutura do Projeto
firedrake-projeto-final-laira/
│
├── src/
│ ├── models/         # Modelos PDE (Firedrake)
│ │ ├── stationary/
│ │ │ ├── dd_model.py # Dirichlet-Dirichlet
│ │ │ ├── dn_model.py # Dirichlet-Neumann
│ │ │
│ │ ├── transient/
│ │ ├── dd_model.py
│ │ ├── dn_model.py
│ │
│ ├── properties/      # Propriedades termodinâmicas
│ │ ├── coolprop.py
│ │ ├── thermo.py
│ │ ├── thermopack.py
│ │
│ ├── interpolation/    # Construção e uso das tabelas
│ │ ├── grid.py         # malha (P,T)
│ │ ├── build_tables.py # cálculo de Z, μ, ρ
│ │ ├── interpolators.py # interpolação (SciPy)
│ │
│ ├── validation/ 
│ │ ├── compare_interp.py # erro da interpolação
│ │ ├── error_analysis.py  (falta criar)
│ │
│ ├── plotting/ 
│ │ ├── plot_properties.py  (vazio)
│ │ ├── plot_results.py     (falta criar)
│ │
│ ├── utils/
│ | ├── constants.py          (vazio)
│ | ├── paths.py
│
├── data/
│ ├── tables/       # tabelas (Z, μ, ρ)
│ ├── processed/            (vazio)
│
├── figures/
│ ├── properties/
│ ├── validation/
│ ├── simulations/
│
├── README.md
├── requirements.txt

    

---

## Modelo Matemático

A equação governante considerada é:

\[
\frac{k}{\mu} \nabla \cdot \left( \frac{p}{Z} \nabla p \right) = 0
\]

onde:
- \( p \): pressão  
- \( k \): permeabilidade  
- \( \mu = \mu(P,T) \): viscosidade  
- \( Z = Z(P,T) \): fator de compressibilidade  

---

## Propriedades Termodinâmicas

As propriedades podem ser obtidas a partir de:

- CoolProp  
- thermo  
- thermopack  

e interpoladas a partir de tabelas bidimensionais:

\[
Z = Z(P,T), \quad \mu = \mu(P,T), \quad \rho = \rho(P,T)
\]

---

## Interpolação

- Tabelas construídas em uma malha (P,T)
- Interpolação via SciPy

---

## Problemas Resolvidos

## casos com condições de contorno Dirichlet-Dirichlet:

Condições de contorno:

- \( p(0) = p_w \)
- \( p(L) = p_r \)

---

## casos com condições de contorno Dirichlet-Neumann:

Condições:

- Inicial: \( p(x,0) = p_r \)
- Contorno: \( p(0,t) = p_w \)


## Casos estacionários:

Resolução:
- Método de Newton (Firedrake)
- Acoplamento não linear via **Iteração de Picard** para μ(P,T) e Z(P,T)


## Casos transientes:

Discretização:
- Elementos CG (grau 1)
- Euler implícito no tempo
- Sistema não linear resolvido via Newton
---

## Estratégia Numérica

O acoplamento das propriedades é feito por:

1. Assume μ e Z conhecidos
2. Resolve a equação de pressão
3. Atualiza μ(P,T) e Z(P,T)
4. Repete até convergência

---

## Execução
Ative o ambiente virtual do Firedrake:

```bash
source ~/venv-firedrake/bin/activate


