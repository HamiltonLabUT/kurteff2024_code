//
//  StartViewController.swift
//  onsetProd4kids
//
//  Created by Garret Kurteff on 1/14/21.
//  Copyright © 2021 Kurteff, Garret. All rights reserved.
//

import UIKit

class StartViewController: UIViewController {

//
// Initialize variables
//
    var version:String = "onsetProd v0.0"
    var currpid:String = "OP0001" // default pid
    var currblock:String = "B1" // default block
    var currset:Bool = true // first 25 or last 25 sentences.
    
//
// Initialize UI variables (@IBOutlets)
//
    @IBOutlet weak var versionLabel: UILabel! // gets our version text
    
//
// UI buttons (@IBActions)
//
// (None as of now)
    
//
// viewDidLoad(), what happens when the Settings view is activated
//
    override func viewDidLoad() {
        super.viewDidLoad()
        let version = versionLabel.text
        print("\(String(format:version!)) successfully loaded.")
    }
    
//
// prepare(), lets us pass values to other view controllers
//
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.destination is ViewController {
            let vc = segue.destination as? ViewController
            vc?.currpid = currpid
            vc?.currblock = currblock
            vc?.currset = currset
        }
    }
}
