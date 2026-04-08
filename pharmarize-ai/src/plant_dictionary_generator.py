#!/usr/bin/env python3
"""
Plant Dictionary Generator for PharmarizeAI
Creates a mapping of local Indonesian plant names to scientific names,
compounds, and medicinal properties.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


# Base plant dictionary with known Indonesian medicinal plants
BASE_PLANTS = [
    {
        "id": "plant_001",
        "local_names": ["pasak bumi", "tongkat ali", "longjack"],
        "scientific_name": "Eurycoma longifolia Jack.",
        "family": "Simaroubaceae",
        "parts_used": ["akar", "kulit batang"],
        "compounds": ["eurycomanone", "eurycomalactone", "eurycomaoside", "eurycomalide A", "niloticin", "longilactone", "quassinoid"],
        "benefits": ["afrodisiaka", "meningkatkan stamina", "antimalaria", "antioksidan"],
        "regions": ["Kalimantan", "Sumatera", "Malaysia"]
    },
    {
        "id": "plant_002",
        "local_names": ["kunyit", "kunir", "turmeric"],
        "scientific_name": "Curcuma longa L.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["kurkumin", "curcumin", "kurkuminoid", "turmerone", "zingiberene"],
        "benefits": ["antioksidan", "antiinflamasi", "hepatoprotektif", "antimikroba"],
        "regions": ["Jawa", "Sumatera", "Bali"]
    },
    {
        "id": "plant_003",
        "local_names": ["jahe", "ginger"],
        "scientific_name": "Zingiber officinale Roscoe",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["gingerol", "shogaol", "zingiberene", "zingerone"],
        "benefits": ["antiemetik", "antiinflamasi", "antioksidan", "menghangatkan tubuh"],
        "regions": ["Jawa", "Sumatera", "Sulawesi"]
    },
    {
        "id": "plant_004",
        "local_names": ["temulawak", "Javanese turmeric"],
        "scientific_name": "Curcuma xanthorrhiza Roxb.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["xanthorrhizol", "kurkumin", "germakron"],
        "benefits": ["hepatoprotektif", "antiinflamasi", "meningkatkan nafsu makan", "antioksidan"],
        "regions": ["Jawa", "Kalimantan"]
    },
    {
        "id": "plant_005",
        "local_names": ["kumis kucing", "cat's whiskers", "java tea"],
        "scientific_name": "Orthosiphon aristatus (Blume) Miq.",
        "family": "Lamiaceae",
        "parts_used": ["daun"],
        "compounds": ["sinensetin", "flavonoid", "saponin", "tanin", "kalium"],
        "benefits": ["diuretik", "antihipertensi", "antidiabetes", "antioksidan"],
        "regions": ["Jawa", "Sumatera", "Kalimantan"]
    },
    {
        "id": "plant_006",
        "local_names": ["sambiloto", "king of bitters", "andrographis"],
        "scientific_name": "Andrographis paniculata (Burm.f.) Nees",
        "family": "Acanthaceae",
        "parts_used": ["daun", "herba"],
        "compounds": ["andrographolide", "neoandrographolide", "flavonoid"],
        "benefits": ["imunomodulator", "antidiabetes", "antiinflamasi", "hepatoprotektif"],
        "regions": ["Jawa", "Sumatera", "Kalimantan"]
    },
    {
        "id": "plant_007",
        "local_names": ["mengkudu", "noni", "pace"],
        "scientific_name": "Morinda citrifolia L.",
        "family": "Rubiaceae",
        "parts_used": ["buah", "daun", "akar"],
        "compounds": ["scopoletin", "antrakuinon", "terpenoid", "xeronine"],
        "benefits": ["antihipertensi", "antioksidan", "antikanker", "imunomodulator"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_008",
        "local_names": ["mahkota dewa", "crown of god"],
        "scientific_name": "Phaleria macrocarpa (Scheff.) Boerl.",
        "family": "Thymelaeaceae",
        "parts_used": ["buah", "daun"],
        "compounds": ["flavonoid", "alkaloid", "saponin", "polifenol", "mahkoside"],
        "benefits": ["antikanker", "antidiabetes", "antiinflamasi", "antihipertensi"],
        "regions": ["Papua", "Irian Jaya"]
    },
    {
        "id": "plant_009",
        "local_names": ["sirsak", "daun sirsak", "soursop", "graviola"],
        "scientific_name": "Annona muricata L.",
        "family": "Annonaceae",
        "parts_used": ["daun", "buah", "biji"],
        "compounds": ["acetogenin", "annonacin", "flavonoid", "alkaloid"],
        "benefits": ["antikanker", "antioksidan", "antiinflamasi", "antidiabetes"],
        "regions": ["Jawa", "Sumatera", "Sulawesi"]
    },
    {
        "id": "plant_010",
        "local_names": ["pegagan", "gotu kola", "antanan"],
        "scientific_name": "Centella asiatica (L.) Urb.",
        "family": "Apiaceae",
        "parts_used": ["herba", "daun"],
        "compounds": ["asiaticoside", "madecassoside", "asiatic acid", "triterpenoid"],
        "benefits": ["nootropik", "penyembuhan luka", "antioksidan", "antiinflamasi"],
        "regions": ["Jawa", "Sumatera", "Bali"]
    },
    {
        "id": "plant_011",
        "local_names": ["bajakah", "bajakah tampala", "bajakah kalalawit"],
        "scientific_name": "Spatholobus littoralis Hassk.",
        "family": "Fabaceae",
        "parts_used": ["batang", "akar"],
        "compounds": ["flavonoid", "tanin", "saponin", "terpenoid", "quercetin"],
        "benefits": ["antikanker", "antioksidan", "antiinflamasi", "penyembuhan luka"],
        "regions": ["Kalimantan Tengah", "Kalimantan"]
    },
    {
        "id": "plant_012",
        "local_names": ["kelor", "moringa", "merunggai"],
        "scientific_name": "Moringa oleifera Lam.",
        "family": "Moringaceae",
        "parts_used": ["daun", "biji", "kulit batang"],
        "compounds": ["quercetin", "kaempferol", "zeatin", "beta-sitosterol"],
        "benefits": ["antioksidan", "antidiabetes", "antiinflamasi", "nutrisi tinggi"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_013",
        "local_names": ["lidah buaya", "aloe vera"],
        "scientific_name": "Aloe vera (L.) Burm.f.",
        "family": "Asphodelaceae",
        "parts_used": ["gel daun", "daun"],
        "compounds": ["aloin", "aloe-emodin", "acemannan", "polisakarida"],
        "benefits": ["penyembuhan luka", "antiinflamasi", "melembabkan kulit", "pencahar"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_014",
        "local_names": ["binahong", "heartleaf madeiravine"],
        "scientific_name": "Anredera cordifolia (Ten.) Steenis",
        "family": "Basellaceae",
        "parts_used": ["daun"],
        "compounds": ["flavonoid", "saponin", "alkaloid", "terpenoid", "asam askorbat"],
        "benefits": ["penyembuhan luka", "antiinflamasi", "antibakteri", "antioksidan"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_015",
        "local_names": ["sirih", "sirih hijau", "betel leaf"],
        "scientific_name": "Piper betle L.",
        "family": "Piperaceae",
        "parts_used": ["daun"],
        "compounds": ["chavicol", "eugenol", "hydroxychavicol", "allylpyrocatechol"],
        "benefits": ["antibakteri", "antijamur", "antioksidan", "menghilangkan bau mulut"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_016",
        "local_names": ["kayu manis", "cinnamon"],
        "scientific_name": "Cinnamomum verum J.Presl",
        "family": "Lauraceae",
        "parts_used": ["kulit batang"],
        "compounds": ["cinnamaldehyde", "eugenol", "coumarin", "polifenol"],
        "benefits": ["antidiabetes", "antioksidan", "antimikroba", "antiinflamasi"],
        "regions": ["Sumatera", "Jawa"]
    },
    {
        "id": "plant_017",
        "local_names": ["cengkeh", "clove"],
        "scientific_name": "Syzygium aromaticum (L.) Merr. & L.M.Perry",
        "family": "Myrtaceae",
        "parts_used": ["bunga", "minyak"],
        "compounds": ["eugenol", "acetyl eugenol", "beta-caryophyllene"],
        "benefits": ["analgesik", "antibakteri", "antiseptik", "antioksidan"],
        "regions": ["Maluku", "Sulawesi"]
    },
    {
        "id": "plant_018",
        "local_names": ["serai", "sereh", "lemongrass"],
        "scientific_name": "Cymbopogon citratus (DC.) Stapf",
        "family": "Poaceae",
        "parts_used": ["batang", "daun"],
        "compounds": ["citral", "geraniol", "citronellal", "limonene"],
        "benefits": ["antimikroba", "antijamur", "antioksidan", "relaksan"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_019",
        "local_names": ["jintan hitam", "habbatussauda", "black seed"],
        "scientific_name": "Nigella sativa L.",
        "family": "Ranunculaceae",
        "parts_used": ["biji"],
        "compounds": ["thymoquinone", "thymohydroquinone", "nigellone", "alpha-pinene"],
        "benefits": ["imunomodulator", "antiinflamasi", "antioksidan", "antidiabetes"],
        "regions": ["diimpor, dibudidayakan di Indonesia"]
    },
    {
        "id": "plant_020",
        "local_names": ["rosella", "roselle"],
        "scientific_name": "Hibiscus sabdariffa L.",
        "family": "Malvaceae",
        "parts_used": ["kelopak bunga"],
        "compounds": ["antosianin", "flavonoid", "asam hibiscus", "polifenol"],
        "benefits": ["antihipertensi", "antioksidan", "hepatoprotektif", "diuretik"],
        "regions": ["Jawa", "Sumatera", "Sulawesi"]
    },
    {
        "id": "plant_021",
        "local_names": ["keji beling", "pecah beling"],
        "scientific_name": "Strobilanthes crispus Blume",
        "family": "Acanthaceae",
        "parts_used": ["daun"],
        "compounds": ["flavonoid", "saponin", "polifenol", "kalium", "silika"],
        "benefits": ["diuretik", "antidiabetes", "meluruhkan batu ginjal", "antioksidan"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_022",
        "local_names": ["meniran", "phyllanthus"],
        "scientific_name": "Phyllanthus niruri L.",
        "family": "Phyllanthaceae",
        "parts_used": ["herba"],
        "compounds": ["filantin", "hipofilantin", "flavonoid", "tanin"],
        "benefits": ["hepatoprotektif", "diuretik", "antivirus", "imunomodulator"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_023",
        "local_names": ["tapak dara", "vinca", "periwinkle"],
        "scientific_name": "Catharanthus roseus (L.) G.Don",
        "family": "Apocynaceae",
        "parts_used": ["daun", "herba"],
        "compounds": ["vincristine", "vinblastine", "alkaloid vinca"],
        "benefits": ["antikanker", "antidiabetes", "antihipertensi"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_024",
        "local_names": ["brotowali", "bratawali", "tinospora"],
        "scientific_name": "Tinospora crispa (L.) Hook. f. & Thomson",
        "family": "Menispermaceae",
        "parts_used": ["batang"],
        "compounds": ["berberin", "tinokrisposid", "alkaloid", "diterpenoid"],
        "benefits": ["antidiabetes", "antipiretik", "imunomodulator", "antimalarial"],
        "regions": ["Jawa", "Kalimantan"]
    },
    {
        "id": "plant_025",
        "local_names": ["temu kunci", "fingerroot", "krachai"],
        "scientific_name": "Boesenbergia rotunda (L.) Mansf.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["pinostrobin", "panduratin A", "flavonoid"],
        "benefits": ["antibakteri", "antijamur", "antioksidan", "afrodisiaka"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_026",
        "local_names": ["lengkuas", "laos", "galangal"],
        "scientific_name": "Alpinia galanga (L.) Willd.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["1'-acetoxychavicol acetate", "galangin", "alpinetin"],
        "benefits": ["antibakteri", "antijamur", "antioksidan", "antiinflamasi"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_027",
        "local_names": ["kencur", "aromatic ginger"],
        "scientific_name": "Kaempferia galanga L.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["ethyl p-methoxycinnamate", "kaempferol", "borneol"],
        "benefits": ["antiinflamasi", "analgesik", "ekspektoran", "antimikroba"],
        "regions": ["Jawa", "Sumatera", "Bali"]
    },
    {
        "id": "plant_028",
        "local_names": ["kapulaga", "cardamom"],
        "scientific_name": "Elettaria cardamomum (L.) Maton",
        "family": "Zingiberaceae",
        "parts_used": ["biji", "buah"],
        "compounds": ["1,8-cineole", "alpha-terpinyl acetate", "limonene"],
        "benefits": ["karminatif", "antiemetik", "antimikroba", "antioksidan"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_029",
        "local_names": ["daun salam", "Indonesian bay leaf"],
        "scientific_name": "Syzygium polyanthum (Wight) Walp.",
        "family": "Myrtaceae",
        "parts_used": ["daun"],
        "compounds": ["eugenol", "flavonoid", "tanin", "minyak atsiri"],
        "benefits": ["antidiabetes", "antihipertensi", "antimikroba", "antioksidan"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_030",
        "local_names": ["daun katuk", "katuk", "sweet leaf"],
        "scientific_name": "Sauropus androgynus (L.) Merr.",
        "family": "Phyllanthaceae",
        "parts_used": ["daun"],
        "compounds": ["papaverine", "flavonoid", "saponin", "asam amino"],
        "benefits": ["laktagog", "antioksidan", "nutrisi tinggi"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_031",
        "local_names": ["daun jambu biji", "jambu biji", "guava"],
        "scientific_name": "Psidium guajava L.",
        "family": "Myrtaceae",
        "parts_used": ["daun", "buah"],
        "compounds": ["quercetin", "guaijaverin", "tanin", "vitamin C"],
        "benefits": ["antidiare", "antioksidan", "antidiabetes", "antimikroba"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_032",
        "local_names": ["lempuyang", "lempuyang emprit"],
        "scientific_name": "Zingiber zerumbet (L.) Roscoe ex Sm.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["zerumbone", "humulene", "caryophyllene"],
        "benefits": ["antiinflamasi", "antioksidan", "analgesik", "antikanker"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_033",
        "local_names": ["kemukus", "cubeb"],
        "scientific_name": "Piper cubeba L.f.",
        "family": "Piperaceae",
        "parts_used": ["buah"],
        "compounds": ["cubebin", "cubebene", "sabinene", "piperine"],
        "benefits": ["antibakteri", "ekspektoran", "diuretik", "karminatif"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_034",
        "local_names": ["bunga telang", "butterfly pea", "kembang telang"],
        "scientific_name": "Clitoria ternatea L.",
        "family": "Fabaceae",
        "parts_used": ["bunga"],
        "compounds": ["antosianin", "delphinidin", "kaempferol", "flavonoid"],
        "benefits": ["antioksidan", "nootropik", "antiinflamasi", "pewarna alami"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_035",
        "local_names": ["adas", "fennel"],
        "scientific_name": "Foeniculum vulgare Mill.",
        "family": "Apiaceae",
        "parts_used": ["biji", "daun"],
        "compounds": ["anethole", "fenchone", "estragole"],
        "benefits": ["karminatif", "laktagog", "ekspektoran", "antispasmodik"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_036",
        "local_names": ["kulit manggis", "manggis", "mangosteen"],
        "scientific_name": "Garcinia mangostana L.",
        "family": "Clusiaceae",
        "parts_used": ["kulit buah"],
        "compounds": ["xanthone", "alpha-mangostin", "gamma-mangostin", "tanin"],
        "benefits": ["antioksidan", "antiinflamasi", "antikanker", "antibakteri"],
        "regions": ["Sumatera", "Kalimantan", "Sulawesi"]
    },
    {
        "id": "plant_037",
        "local_names": ["pandan", "pandan wangi", "screwpine"],
        "scientific_name": "Pandanus amaryllifolius Roxb.",
        "family": "Pandanaceae",
        "parts_used": ["daun"],
        "compounds": ["2-acetyl-1-pyrroline", "alkaloid", "flavonoid"],
        "benefits": ["pewangi alami", "antidiabetes", "antioksidan"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_038",
        "local_names": ["daun pepaya", "pepaya"],
        "scientific_name": "Carica papaya L.",
        "family": "Caricaceae",
        "parts_used": ["daun", "buah", "getah"],
        "compounds": ["papain", "carpaine", "flavonoid", "alkaloid"],
        "benefits": ["antimalarial", "antidiabetes", "meningkatkan trombosit", "pencernaan"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_039",
        "local_names": ["daun dewa", "beluntas cina"],
        "scientific_name": "Gynura pseudochina (L.) DC.",
        "family": "Asteraceae",
        "parts_used": ["daun"],
        "compounds": ["flavonoid", "saponin", "tanin", "steroid"],
        "benefits": ["antidiabetes", "antihipertensi", "antiinflamasi"],
        "regions": ["Jawa"]
    },
    {
        "id": "plant_040",
        "local_names": ["keladi tikus", "typhonium"],
        "scientific_name": "Typhonium flagelliforme (Lodd.) Blume",
        "family": "Araceae",
        "parts_used": ["umbi", "daun"],
        "compounds": ["flavonoid", "alkaloid", "steroid", "glikosida"],
        "benefits": ["antikanker", "antiinflamasi", "detoksifikasi"],
        "regions": ["Jawa", "Sumatera"]
    },
    # Additional plants found in journals
    {
        "id": "plant_041",
        "local_names": ["akar kuning", "kayu kuning"],
        "scientific_name": "Arcangelisia flava (L.) Merr.",
        "family": "Menispermaceae",
        "parts_used": ["batang", "akar"],
        "compounds": ["berberin", "palmatine", "jatrorrhizine"],
        "benefits": ["antimikroba", "antidiabetes", "antiinflamasi"],
        "regions": ["Kalimantan", "Sumatera"]
    },
    {
        "id": "plant_042",
        "local_names": ["saga", "saga telik", "crab's eye"],
        "scientific_name": "Abrus precatorius L.",
        "family": "Fabaceae",
        "parts_used": ["daun", "biji"],
        "compounds": ["abrin", "glycyrrhizin", "abrusoside"],
        "benefits": ["antiinflamasi", "ekspektoran", "antimikroba"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_043",
        "local_names": ["jeruk nipis", "lime"],
        "scientific_name": "Citrus aurantifolia (Christm.) Swingle",
        "family": "Rutaceae",
        "parts_used": ["buah", "kulit", "daun"],
        "compounds": ["limonene", "citric acid", "hesperidin", "vitamin C"],
        "benefits": ["antioksidan", "antimikroba", "meningkatkan imunitas"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_044",
        "local_names": ["belimbing wuluh", "belimbing sayur"],
        "scientific_name": "Averrhoa bilimbi L.",
        "family": "Oxalidaceae",
        "parts_used": ["buah", "daun"],
        "compounds": ["oxalic acid", "flavonoid", "saponin"],
        "benefits": ["antihipertensi", "antidiabetes", "antiinflamasi"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_045",
        "local_names": ["pinang", "betel nut"],
        "scientific_name": "Areca catechu L.",
        "family": "Arecaceae",
        "parts_used": ["biji"],
        "compounds": ["arecoline", "arecaidine", "tanin", "flavonoid"],
        "benefits": ["anthelmintik", "stimulan", "astringen"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_046",
        "local_names": ["kenanga", "cananga"],
        "scientific_name": "Cananga odorata (Lam.) Hook.f. & Thomson",
        "family": "Annonaceae",
        "parts_used": ["bunga"],
        "compounds": ["linalool", "benzyl acetate", "geraniol", "ylang-ylang oil"],
        "benefits": ["aromaterapi", "antiseptik", "relaksan"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_047",
        "local_names": ["senggani", "senduduk"],
        "scientific_name": "Melastoma malabathricum L.",
        "family": "Melastomataceae",
        "parts_used": ["daun", "buah"],
        "compounds": ["flavonoid", "tanin", "saponin", "antosianin"],
        "benefits": ["antidiare", "penyembuhan luka", "antioksidan"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_048",
        "local_names": ["daun ungu", "handeleum"],
        "scientific_name": "Graptophyllum pictum (L.) Griff.",
        "family": "Acanthaceae",
        "parts_used": ["daun"],
        "compounds": ["flavonoid", "steroid", "tanin", "alkaloid"],
        "benefits": ["antihemoroid", "antiinflamasi", "pencahar"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_049",
        "local_names": ["bawang dayak", "bawang tiwai"],
        "scientific_name": "Eleutherine palmifolia (L.) Merr.",
        "family": "Iridaceae",
        "parts_used": ["umbi"],
        "compounds": ["eleutherinol", "antrakuinon", "naftokuinon"],
        "benefits": ["antikanker", "antimikroba", "antioksidan"],
        "regions": ["Kalimantan"]
    },
    {
        "id": "plant_050",
        "local_names": ["pulai", "devil tree"],
        "scientific_name": "Alstonia scholaris (L.) R.Br.",
        "family": "Apocynaceae",
        "parts_used": ["kulit batang"],
        "compounds": ["alstonine", "echitamine", "alkaloid indol"],
        "benefits": ["antimalaria", "antipiretik", "antidiare"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_051",
        "local_names": ["rumput mutiara"],
        "scientific_name": "Hedyotis corymbosa (L.) Lam.",
        "family": "Rubiaceae",
        "parts_used": ["herba"],
        "compounds": ["asperuloside", "oleanolic acid", "ursolic acid"],
        "benefits": ["antikanker", "antiinflamasi", "hepatoprotektif"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_052",
        "local_names": ["jengkol", "jering"],
        "scientific_name": "Archidendron pauciflorum (Benth.) I.C.Nielsen",
        "family": "Fabaceae",
        "parts_used": ["biji", "kulit"],
        "compounds": ["jengkolic acid", "flavonoid", "saponin"],
        "benefits": ["antidiabetes", "antimikroba"],
        "regions": ["Jawa", "Sumatera"]
    },
    {
        "id": "plant_053",
        "local_names": ["petai", "pete", "bitter bean"],
        "scientific_name": "Parkia speciosa Hassk.",
        "family": "Fabaceae",
        "parts_used": ["biji"],
        "compounds": ["thiazolidine-4-carboxylic acid", "flavonoid", "saponin"],
        "benefits": ["antidiabetes", "antioksidan", "antihipertensi"],
        "regions": ["seluruh Indonesia"]
    },
    {
        "id": "plant_054",
        "local_names": ["temu hitam", "black turmeric"],
        "scientific_name": "Curcuma aeruginosa Roxb.",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["kurkumin", "zedoary", "germakron"],
        "benefits": ["anthelmintik", "antiinflamasi", "meningkatkan nafsu makan"],
        "regions": ["Jawa"]
    },
    {
        "id": "plant_055",
        "local_names": ["temu putih", "zedoary"],
        "scientific_name": "Curcuma zedoaria (Christm.) Roscoe",
        "family": "Zingiberaceae",
        "parts_used": ["rimpang"],
        "compounds": ["kurkumin", "kurkumenol", "germakron"],
        "benefits": ["antioksidan", "hepatoprotektif", "antiinflamasi"],
        "regions": ["Jawa", "Sumatera"]
    },
]


def extract_plants_from_journals(processed_dir: str) -> List[Dict]:
    """Extract additional plant mentions from processed journals."""
    processed_path = Path(processed_dir)
    txt_files = list(processed_path.glob("*.txt"))
    
    # Pattern to find plant names with scientific names in parentheses
    plant_pattern = re.compile(
        r'([a-z]+(?:\s+[a-z]+)?)\s*\(([A-Z][a-z]+\s+[a-z]+(?:\s+[A-Z][a-z\.]+)?)\)',
        re.IGNORECASE
    )
    
    found_plants = defaultdict(lambda: {"count": 0, "scientific_names": set()})
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        matches = plant_pattern.findall(text)
        for local_name, scientific_name in matches:
            local_lower = local_name.lower().strip()
            # Skip common non-plant words
            skip_words = {'ekstrak', 'metode', 'hasil', 'penelitian', 'uji', 'aktivitas'}
            if local_lower in skip_words:
                continue
            if len(local_lower) >= 3:
                found_plants[local_lower]["count"] += 1
                found_plants[local_lower]["scientific_names"].add(scientific_name)
    
    return found_plants


def build_dictionary(base_plants: List[Dict], processed_dir: str) -> Dict:
    """Build the complete plant dictionary."""
    # Start with base plants
    dictionary = {
        "version": "1.0",
        "description": "Kamus tanaman obat Indonesia untuk Pharmarize.ai",
        "total_plants": len(base_plants),
        "plants": base_plants,
        "local_name_index": {},
        "scientific_name_index": {},
        "compound_index": defaultdict(list),
        "benefit_index": defaultdict(list),
    }
    
    # Build indices
    for plant in base_plants:
        plant_id = plant["id"]
        
        # Index by local names
        for local_name in plant["local_names"]:
            dictionary["local_name_index"][local_name.lower()] = plant_id
        
        # Index by scientific name
        sci_name = plant["scientific_name"].split()[0:2]
        sci_key = " ".join(sci_name).lower()
        dictionary["scientific_name_index"][sci_key] = plant_id
        
        # Index by compounds
        for compound in plant.get("compounds", []):
            dictionary["compound_index"][compound.lower()].append(plant_id)
        
        # Index by benefits
        for benefit in plant.get("benefits", []):
            dictionary["benefit_index"][benefit.lower()].append(plant_id)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    dictionary["compound_index"] = dict(dictionary["compound_index"])
    dictionary["benefit_index"] = dict(dictionary["benefit_index"])
    
    return dictionary


def save_dictionary(dictionary: Dict, output_path: str):
    """Save the dictionary to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    
    print(f"Dictionary saved to: {output_file}")
    print(f"  Total plants: {dictionary['total_plants']}")
    print(f"  Local name entries: {len(dictionary['local_name_index'])}")
    print(f"  Scientific name entries: {len(dictionary['scientific_name_index'])}")
    print(f"  Compound entries: {len(dictionary['compound_index'])}")
    print(f"  Benefit entries: {len(dictionary['benefit_index'])}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    processed_dir = script_dir / "data" / "processed"
    output_path = script_dir / "data" / "plant_dictionary.json"
    
    print("Building plant dictionary...")
    dictionary = build_dictionary(BASE_PLANTS, str(processed_dir))
    save_dictionary(dictionary, str(output_path))
