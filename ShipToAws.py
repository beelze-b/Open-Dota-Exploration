
# coding: utf-8

# In[ ]:

import requests
import json


# In[ ]:

heroInfo = {
        "heroes":[
        {
            "name":"npc_dota_hero_antimage",
                "id":1
        },
        {
            "name":"npc_dota_hero_axe",
            "id":2
        },
        {
            "name":"npc_dota_hero_bane",
            "id":3
        },
        {
            "name":"npc_dota_hero_bloodseeker",
            "id":4
        },
        {
            "name":"npc_dota_hero_crystal_maiden",
            "id":5
        },
        {
            "name":"npc_dota_hero_drow_ranger",
            "id":6
        },
        {
            "name":"npc_dota_hero_earthshaker",
            "id":7
        },
        {
            "name":"npc_dota_hero_juggernaut",
            "id":8
        },
        {
            "name":"npc_dota_hero_mirana",
            "id":9
        },
        {
            "name":"npc_dota_hero_nevermore",
            "id":11
        },
        {
            "name":"npc_dota_hero_morphling",
            "id":10
        },
        {
            "name":"npc_dota_hero_phantom_lancer",
            "id":12
        },
        {
            "name":"npc_dota_hero_puck",
            "id":13
        },
        {
            "name":"npc_dota_hero_pudge",
            "id":14
        },
        {
            "name":"npc_dota_hero_razor",
            "id":15
        },
        {
            "name":"npc_dota_hero_sand_king",
            "id":16
        },
        {
            "name":"npc_dota_hero_storm_spirit",
            "id":17
        },
        {
            "name":"npc_dota_hero_sven",
            "id":18
        },
        {
            "name":"npc_dota_hero_tiny",
            "id":19
        },
        {
            "name":"npc_dota_hero_vengefulspirit",
            "id":20
        },
        {
            "name":"npc_dota_hero_windrunner",
            "id":21
        },
        {
            "name":"npc_dota_hero_zuus",
            "id":22
        },
        {
            "name":"npc_dota_hero_kunkka",
            "id":23
        },
        {
            "name":"npc_dota_hero_lina",
            "id":25
        },
        {
            "name":"npc_dota_hero_lich",
            "id":31
        },
        {
            "name":"npc_dota_hero_lion",
            "id":26
        },
        {
            "name":"npc_dota_hero_shadow_shaman",
            "id":27
        },
        {
            "name":"npc_dota_hero_slardar",
            "id":28
        },
        {
            "name":"npc_dota_hero_tidehunter",
            "id":29
        },
        {
            "name":"npc_dota_hero_witch_doctor",
            "id":30
        },
        {
            "name":"npc_dota_hero_riki",
            "id":32
        },
        {
            "name":"npc_dota_hero_enigma",
            "id":33
        },
        {
            "name":"npc_dota_hero_tinker",
            "id":34
        },
        {
            "name":"npc_dota_hero_sniper",
            "id":35
        },
        {
            "name":"npc_dota_hero_necrolyte",
            "id":36
        },
        {
            "name":"npc_dota_hero_warlock",
            "id":37
        },
        {
            "name":"npc_dota_hero_beastmaster",
            "id":38
        },
        {
            "name":"npc_dota_hero_queenofpain",
            "id":39
        },
        {
            "name":"npc_dota_hero_venomancer",
            "id":40
        },
        {
            "name":"npc_dota_hero_faceless_void",
            "id":41
        },
        {
            "name":"npc_dota_hero_skeleton_king",
            "id":42
        },
        {
            "name":"npc_dota_hero_death_prophet",
            "id":43
        },
        {
            "name":"npc_dota_hero_phantom_assassin",
            "id":44
        },
        {
            "name":"npc_dota_hero_pugna",
            "id":45
        },
        {
            "name":"npc_dota_hero_templar_assassin",
            "id":46
        },
        {
            "name":"npc_dota_hero_viper",
            "id":47
        },
        {
            "name":"npc_dota_hero_luna",
            "id":48
        },
        {
            "name":"npc_dota_hero_dragon_knight",
            "id":49
        },
        {
            "name":"npc_dota_hero_dazzle",
            "id":50
        },
        {
            "name":"npc_dota_hero_rattletrap",
            "id":51
        },
        {
            "name":"npc_dota_hero_leshrac",
            "id":52
        },
        {
            "name":"npc_dota_hero_furion",
            "id":53
        },
        {
            "name":"npc_dota_hero_life_stealer",
            "id":54
        },
        {
            "name":"npc_dota_hero_dark_seer",
            "id":55
        },
        {
            "name":"npc_dota_hero_clinkz",
            "id":56
        },
        {
            "name":"npc_dota_hero_omniknight",
            
            "id":57
        },
        {
            "name":"npc_dota_hero_enchantress",
            "id":58
        },
        {
            "name":"npc_dota_hero_huskar",
            "id":59
        },
        {
            "name":"npc_dota_hero_night_stalker",
            "id":60
        },
        {
            "name":"npc_dota_hero_broodmother",
            "id":61
        },
        {
            "name":"npc_dota_hero_bounty_hunter",
            "id":62
        },
        {
            "name":"npc_dota_hero_weaver",
            "id":63
        },
        {
            "name":"npc_dota_hero_jakiro",
            "id":64
        },
        {
            "name":"npc_dota_hero_batrider",
            "id":65
        },
        {
            "name":"npc_dota_hero_chen",
            "id":66
        },
        {
            "name":"npc_dota_hero_spectre",
            "id":67
        },
        {
            "name":"npc_dota_hero_doom_bringer",
            "id":69
        },
        {
            "name":"npc_dota_hero_ancient_apparition",
            "id":68
        },
        {
            "name":"npc_dota_hero_ursa",
            "id":70
        },
        {
            "name":"npc_dota_hero_spirit_breaker",
            "id":71
        },
        {
            "name":"npc_dota_hero_gyrocopter",
            "id":72
        },
        {
            "name":"npc_dota_hero_alchemist",
            "id":73
        },
        {
            "name":"npc_dota_hero_invoker",
            "id":74
        },
        {
            "name":"npc_dota_hero_silencer",
            "id":75
        },
        {
            "name":"npc_dota_hero_obsidian_destroyer",
            "id":76
        },
        {
            "name":"npc_dota_hero_lycan",
            "id":77
        },
        {
            "name":"npc_dota_hero_brewmaster",
            "id":78
        },
        {
            "name":"npc_dota_hero_shadow_demon",
            "id":79
        },
        {
            "name":"npc_dota_hero_lone_druid",
            "id":80
        },
        {
            "name":"npc_dota_hero_chaos_knight",
            "id":81
        },
        {
            "name":"npc_dota_hero_meepo",
            "id":82
        },
        {
            "name":"npc_dota_hero_treant",
            "id":83
        },
        {
            "name":"npc_dota_hero_ogre_magi",
            "id":84
        },
        {
            "name":"npc_dota_hero_undying",
            "id":85
        },
        {
            "name":"npc_dota_hero_rubick",
            "id":86
        },
        {
            "name":"npc_dota_hero_disruptor",
            "id":87
        },
        {
            "name":"npc_dota_hero_nyx_assassin",
            "id":88
        },
        {
            "name":"npc_dota_hero_naga_siren",
            "id":89
        },
        {
            "name":"npc_dota_hero_keeper_of_the_light",
            "id":90
        },
        {
            "name":"npc_dota_hero_wisp",
            "id":91
        },
        {
            "name":"npc_dota_hero_visage",
            "id":92
        },
        {
            "name":"npc_dota_hero_slark",
            "id":93
        },
        {
            "name":"npc_dota_hero_medusa",
            "id":94
        },
        {
            "name":"npc_dota_hero_troll_warlord",
            "id":95
        },
        {
            "name":"npc_dota_hero_centaur",
            "id":96
        },
        {
            "name":"npc_dota_hero_magnataur",
            "id":97
        },
        {
            "name":"npc_dota_hero_shredder",
            "id":98
        },
        {
            "name":"npc_dota_hero_bristleback",
            "id":99
        },
        {
            "name":"npc_dota_hero_tusk",
            "id":100
        },
        {
            "name":"npc_dota_hero_skywrath_mage",
            "id":101
        },
        {
            "name":"npc_dota_hero_abaddon",
            "id":102
        },
        {
            "name":"npc_dota_hero_elder_titan",
            "id":103
        },
        {
            "name":"npc_dota_hero_legion_commander",
            "id":104
        },
        {
            "name":"npc_dota_hero_ember_spirit",
            "id":106
        },
        {
            "name":"npc_dota_hero_earth_spirit",
            "id":107
        },
        {
            "name":"npc_dota_hero_terrorblade",
            "id":109
        },
        {
            "name":"npc_dota_hero_phoenix",
            "id":110
        },
        {
            "name":"npc_dota_hero_oracle",
            "id":111
        },
        {
            "name":"npc_dota_hero_techies",
            "id":105
        },
        {
            "name":"npc_dota_hero_winter_wyvern",
            "id":112
        },
        {
            "name":"npc_dota_hero_arc_warden",
            "id":113
        },
        {
            "name":"npc_dota_hero_abyssal_underlord",
            "id":108
        },
        {
            "name":"npc_dota_hero_monkey_king",
            "id":114
        }
        ]
}

heroRoles = {
    1: [1], 2: [3, 2, 4], 3: [5, 4],
    4: [3, 2], 5: [5, 4], 6: [1, 2],
    7: [4, 5], 8: [1, 2], 9: [2, 3, 1],
    10: [1, 2], 11: [2, 1], 12: [1],
    13: [2], 14: [2, 4], 15: [2],
    16: [3, 4], 17: [2], 18: [1],
    19: [2, 4], 20: [5, 4, 1], 21: [2],
    22: [2], 23: [2, 3], 25: [2, 4],
    26: [4, 5], 27: [5, 4, 2], 28: [4, 3, 1],
    29: [3, 4], 30: [4, 5], 31: [5],
    32: [4], 33: [4], 34: [2],
    35: [2, 1, 3], 36: [3, 4, 2], 37: [4, 5],
    38: [3, 4], 39: [2], 40: [4, 2],
    41: [3, 1], 42: [3, 1], 43: [2],
    44: [1], 45: [4, 2, 5], 46: [2],
    47: [2], 48: [1], 49: [2, 3, 1],
    50: [5, 4], 51: [3], 52: [2, 4],
    53: [4], 54: [4, 3], 55: [3],
    56: [3], 57: [5], 58: [4],
    59: [2, 1, 3], 60: [4, 2], 61: [3],
    62: [4], 63: [3], 64: [5, 2],
    65: [2, 3], 66: [4], 67: [1],
    68: [5], 69: [4, 3], 70: [4],
    71: [4, 3], 72: [1], 73: [2, 4],
    74: [2], 75: [5], 76: [2],
    77: [4], 78: [2], 79: [5],
    80: [4, 3], 81: [1, 4], 82: [2],
    83: [5], 84: [5], 85: [5, 4],
    86: [5], 87: [5], 88: [3],
    89: [1, 4], 90: [5], 91: [5],
    92: [4], 93: [1], 94: [1], 
    95: [1], 96: [3], 97: [2],
    98: [3, 2], 99: [3], 100: [3],
    101: [5, 4], 102: [5], 103: [3],
    104: [4], 105: [4], 106: [2],
    107: [4], 108: [3], 109: [1],
    110: [4], 111: [5], 112: [5],
    113: [2], 114: [2, 4]   
}


# In[ ]:

import random
# return the roles of farm priority (1-5) for the corresponding hero ID
# the roles are in descending order of common if there is more than one
def classifyHeroRoles(heroID):
    return heroRoles[heroID]

def giveHeroRole(heroID, availableRoles):
    roles = classifyHeroRoles(heroID)
    if len(roles) == 1 and roles[0] in availableRoles:
        return roles[0]
    elif len(roles) == 1:
        return random.choice(availableRoles)
    for role in roles:
        if role in availableRoles:
            return role
    return random.choice(availableRoles)

def classifyTeam(heroIDsOnFullTeam):
    classifiedTeam = {}
    availableRoles = [1, 2, 3, 4, 5]
    heroIDsOnFullTeam = sorted(heroIDsOnFullTeam, key = lambda x: len(classifyHeroRoles(x)))
    for heroID in heroIDsOnFullTeam:
        givenRole = giveHeroRole(heroID, availableRoles)
        availableRoles.remove(givenRole)
        classifiedTeam[givenRole] = heroID
    return classifiedTeam
    


# In[ ]:

def PlayerExtractKey(playerJSON, key):
    if key in playerJSON:
        return playerJSON[key]
    else:
        return float('nan')

def extractDotaInformation(matchJSON):
    matchInfo = {}
    matchInfo['match_id'] = matchJSON['match_id']
    matchInfo['patch'] = matchJSON['patch']
    matchInfo['positive_votes'] = matchJSON['positive_votes']
    matchInfo['negative_votes'] = matchJSON['negative_votes']
    matchInfo['first_blood_time'] = matchJSON['first_blood_time']
    matchInfo['barracks_status_radiant'] = matchJSON['barracks_status_radiant']
    matchInfo['barracks_status_dire'] = matchJSON['barracks_status_dire']
    matchInfo['tower_status_radiant'] = matchJSON['tower_status_radiant']
    matchInfo['tower_status_dire'] = matchJSON['tower_status_dire']
    matchInfo['radiant_win'] = matchJSON['radiant_win']
    matchInfo['duration'] = matchJSON['duration']
    
    heroesRadiant = []
    heroesDire = []
    
    for player in matchJSON['players']:
        if player['isRadiant']:
            heroesRadiant.append(player['hero_id'])
        else:
            heroesDire.append(player['hero_id'])
            
    heroesRadiant = classifyTeam(heroesRadiant)
    heroesDire = classifyTeam(heroesDire)
    matchInfo['radiant_pos1'] = heroesRadiant[1]
    matchInfo['radiant_pos2'] = heroesRadiant[2]
    matchInfo['radiant_pos3'] = heroesRadiant[3]
    matchInfo['radiant_pos4'] = heroesRadiant[4]
    matchInfo['radiant_pos5'] = heroesRadiant[5]
    matchInfo['dire_pos1'] = heroesDire[1]
    matchInfo['dire_pos2'] = heroesDire[2]
    matchInfo['dire_pos3'] = heroesDire[3]
    matchInfo['dire_pos4'] = heroesDire[4]
    matchInfo['dire_pos5'] = heroesDire[5]
    for player in matchJSON['players']:
        factions = ['radiant', 'dire']
        poss = ['_pos1', '_pos2', '_pos3', '_pos4', '_pos5']
        for faction in factions:
            for pos in poss:
                if player['hero_id'] == matchInfo[faction+pos]:
                    factionpos = faction+pos
                    matchInfo[factionpos+'_item0'] = PlayerExtractKey(player, 'item_0')
                    matchInfo[factionpos+'_item1'] = PlayerExtractKey(player, 'item_1')
                    matchInfo[factionpos+'_item2'] = PlayerExtractKey(player, 'item_2')
                    matchInfo[factionpos+'_item3'] = PlayerExtractKey(player, 'item_3')
                    matchInfo[factionpos+'_item4'] = PlayerExtractKey(player, 'item_4')
                    matchInfo[factionpos+'_item5'] = PlayerExtractKey(player, 'item_5')
                    matchInfo[factionpos+'_kills'] = PlayerExtractKey(player, 'kills')
                    matchInfo[factionpos+'_deaths'] = PlayerExtractKey(player, 'deaths')
                    matchInfo[factionpos+'_assists'] = PlayerExtractKey(player, 'assists')
                    matchInfo[factionpos+'_apm'] = PlayerExtractKey(player, 'actions_per_min')
                    matchInfo[factionpos+'_kpm'] = PlayerExtractKey(player, 'kills_per_min')                                               
                    matchInfo[factionpos+'_kda'] = PlayerExtractKey(player, 'kda')
                    matchInfo[factionpos+'_hero_dmg'] = PlayerExtractKey(player, 'hero_damage')
                    matchInfo[factionpos+'_gpm'] = PlayerExtractKey(player, 'gold_per_min')
                    matchInfo[factionpos+'_hero_heal'] = PlayerExtractKey(player, 'hero_healing')
                    matchInfo[factionpos+'_xpm'] = PlayerExtractKey(player, 'xp_per_min')
                    matchInfo[factionpos+'_totalgold'] = PlayerExtractKey(player, 'total_gold')
                    matchInfo[factionpos+'_totalxp'] = PlayerExtractKey(player, 'total_xp')
                    matchInfo[factionpos+'_lasthits'] = PlayerExtractKey(player, 'last_hits')
                    matchInfo[factionpos+'_denies'] = PlayerExtractKey(player, 'denies')
                    matchInfo[factionpos+'_tower_kills'] = PlayerExtractKey(player, 'tower_kills')
                    matchInfo[factionpos+'_courier_kills'] = PlayerExtractKey(player, 'courier_kills')
                    matchInfo[factionpos+'_gold_spent'] = PlayerExtractKey(player, 'gold_spent')
                    matchInfo[factionpos+'_observer_uses'] = PlayerExtractKey(player, 'observer_uses')
                    matchInfo[factionpos+'_sentry_uses'] = PlayerExtractKey(player, 'sentry_uses')
                    matchInfo[factionpos+'_ancient_kills'] = PlayerExtractKey(player, 'ancient_kills')
                    matchInfo[factionpos+'_neutral_kills'] = PlayerExtractKey(player, 'neutral_kills')
                    matchInfo[factionpos+'_camps_stacked'] = PlayerExtractKey(player, 'camps_stacked')
                    matchInfo[factionpos+'_pings'] = PlayerExtractKey(player, 'pings')
                    matchInfo[factionpos+'_rune_pickups'] = PlayerExtractKey(player, 'rune_pickups')
                    
    return matchInfo
        


# In[ ]:

from time import sleep
from itertools import chain
from collections import defaultdict
import pandas as pd


def GoThroughABlock(initialMatchID=2976775347, blockOfMatches=100):
    mainDict = {}
    
    for currentMatchID in range(initialMatchID, initialMatchID + blockOfMatches):
        host = "https://api.opendota.com/api/matches/" + str(currentMatchID)
        data = {'match_id': currentMatchID}
        data = requests.get(host, data)
        
        if data.status_code != 200:
            continue
            
        matchJSON = json.loads(data.content)
        
        if 'lobby_type' not in matchJSON:
            continue
        lobby_type = matchJSON['lobby_type']
        # 0 and 7 correspond to normal and ranked
        if lobby_type != 0 and lobby_type != 7 and lobby_type != 1:
            continue
            
        matchPerformance = extractDotaInformation(matchJSON)       
        for k, v in matchPerformance.items():
            if k in mainDict:
                mainDict[k].append(v)
            else: 
                mainDict[k] = [v]
        sleep(0.9)
    return pd.DataFrame.from_dict(mainDict)


# In[ ]:

match_id = 2976775347
block = 100
for _ in range(4000):
    data = GoThroughABlock(match_id, block)
    data.to_csv("data/to_process/match{0}_block{1}.csv".format(match_id, block), index=False)
    match_id = match_id + block

